import argparse
from collections import Counter
import csv
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import tempfile
import threading
import time

KV_BYTES_PER_TOKEN = 2 * 32 * 32 * 128 * 2
TOKEN_BUDGET = 2048
OUTPUT_LEN = 128


class ThermalStop(RuntimeError):
    pass


def cooling_decision(temperature, sw_thermal, hw_thermal, previous_hot):
    hot = temperature >= 88 or sw_thermal == "Active" or hw_thermal == "Active"
    consecutive = previous_hot + 1 if hot else 0
    reason = (
        "temperature_at_least_95C"
        if temperature >= 95
        else "sustained_heat_or_thermal_slowdown" if consecutive >= 3 else None
    )
    return consecutive, reason


def classify(computed_before, prompt_tokens, scheduled):
    prompt_work = max(0, min(scheduled, prompt_tokens - computed_before))
    return prompt_work, scheduled - prompt_work


def checkpoint_dtypes(path):
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        assert 0 < header_size <= 64 * 1024**2, "Invalid safetensors header size"
        header = json.loads(stream.read(header_size))
    counts = Counter()
    for name, tensor in header.items():
        if name != "__metadata__":
            counts[tensor["dtype"]] += math.prod(tensor["shape"])
    return dict(counts)


def self_check():
    assert cooling_decision(95, "Not Active", "Not Active", 0)[1] == "temperature_at_least_95C"
    assert cooling_decision(88, "Not Active", "Not Active", 2)[1] == "sustained_heat_or_thermal_slowdown"
    assert cooling_decision(70, "Active", "Not Active", 2)[1] == "sustained_heat_or_thermal_slowdown"
    assert cooling_decision(70, "Not Active", "Not Active", 2) == (0, None)
    assert KV_BYTES_PER_TOKEN == 524288
    assert KV_BYTES_PER_TOKEN / 1024 == 512
    assert math.ceil((1920 + 127) / 16) * 16 * KV_BYTES_PER_TOKEN == 1_073_741_824
    assert classify(0, 128, 128) == (128, 0)
    assert classify(127, 128, 1) == (1, 0)
    assert classify(128, 128, 1) == (0, 1)
    assert classify(128 + 126, 128, 1) == (0, 1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "header.safetensors"
        header = json.dumps({"__metadata__": {}, "weight": {"dtype": "F16", "shape": [2, 3]}}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header)
        assert checkpoint_dtypes(path) == {"F16": 6}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, default=str) + "\n")
        stream.flush()


def gpu_snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,power.draw,"
            "power.limit,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu",
            "--format=csv",
            "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
        ],
        text=True,
        timeout=15,
    ).strip()


def main(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import vllm
    from pydantic import TypeAdapter
    from vllm import SamplingParams
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    if vllm.__version__ != "0.11.0":
        raise RuntimeError(f"Expected vLLM 0.11.0, got {vllm.__version__}")
    model_dir = args.model.resolve()
    output_dir = args.output.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    before_gpu = gpu_snapshot()
    telemetry_file = (output_dir / "telemetry.csv").open("w")
    thermal_stop = threading.Event()
    shutting_down = threading.Event()
    thermal_state = {
        "samples": 0,
        "consecutive_hot": 0,
        "latest": None,
        "stop_reason": None,
    }
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,uuid,memory.used,memory.total,"
            "utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,"
            "clocks.sm,clocks.mem,pstate,fan.speed,clocks_event_reasons.sw_thermal_slowdown,"
            "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap",
            "--format=csv,nounits",
            "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
            "--loop-ms=1000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def monitor_cooling():
        try:
            header = None
            for line in telemetry.stdout:
                telemetry_file.write(line)
                telemetry_file.flush()
                values = next(csv.reader([line], skipinitialspace=True))
                if header is None:
                    header = values
                    continue
                sample = dict(zip(header, values))
                streak, reason = cooling_decision(
                    float(sample["temperature.gpu"]),
                    sample["clocks_event_reasons.sw_thermal_slowdown"],
                    sample["clocks_event_reasons.hw_thermal_slowdown"],
                    thermal_state["consecutive_hot"],
                )
                thermal_state.update(
                    samples=thermal_state["samples"] + 1,
                    consecutive_hot=streak,
                    latest=sample,
                )
                if reason and not thermal_stop.is_set():
                    thermal_state["stop_reason"] = reason
                    write_json(output_dir / "thermal_stop.json", dict(thermal_state))
                    thermal_stop.set()
            if not shutting_down.is_set():
                raise RuntimeError("Telemetry ended before benchmark cleanup")
        except Exception as error:
            if not shutting_down.is_set():
                thermal_state["stop_reason"] = f"telemetry_failure: {error}"
                write_json(output_dir / "thermal_stop.json", dict(thermal_state))
                thermal_stop.set()

    monitor = threading.Thread(target=monitor_cooling, daemon=True)
    monitor.start()

    def check_cooling():
        if thermal_stop.is_set():
            raise ThermalStop(thermal_state["stop_reason"])

    engine = None
    screen_count = 0
    try:
        kwargs = dict(
            model=str(model_dir),
            tokenizer_mode="auto",
            config_format="auto",
            load_format="safetensors",
            dtype="float16",
            kv_cache_dtype="auto",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            cpu_offload_gb=0,
            max_model_len=2048,
            max_num_seqs=16,
            max_num_batched_tokens=TOKEN_BUDGET,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=TOKEN_BUDGET,
            enable_prefix_caching=False,
            block_size=16,
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.92,
            async_scheduling=False,
            enforce_eager=False,
            compilation_config=CompilationConfig(
                level=0,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            ),
            seed=20261103,
            disable_log_stats=True,
            generation_config="vllm",
        )
        write_json(output_dir / "requested_config.json", kwargs)
        print("LOAD_START", datetime.now(timezone.utc).isoformat(), flush=True)
        engine = LLMEngine.from_engine_args(
            EngineArgs(**kwargs), enable_multiprocessing=False
        )
        check_cooling()
        core = engine.engine_core.engine_core
        scheduler = core.scheduler
        manager = scheduler.kv_cache_manager
        assert (
            core.batch_queue is None
            and not engine.vllm_config.scheduler_config.async_scheduling
        )
        assert not engine.vllm_config.cache_config.enable_prefix_caching
        worker = core.model_executor.driver_worker.worker
        runner = worker.model_runner
        config = engine.vllm_config
        text = config.model_config.hf_text_config
        geometry = {
            k: getattr(text, k, None)
            for k in [
                "num_hidden_layers",
                "num_key_value_heads",
                "head_dim",
                "hidden_size",
                "num_attention_heads",
                "sliding_window",
            ]
        }
        assert geometry["num_hidden_layers"] == 32
        assert geometry["num_key_value_heads"] == 32 and geometry["head_dim"] == 128
        assert geometry["sliding_window"] is None
        assert config.model_config.dtype == torch.float16
        kv_config = scheduler.kv_cache_config
        groups = kv_config.kv_cache_groups
        assert len(groups) == 1
        spec = groups[0].kv_cache_spec
        assert len(groups[0].layer_names) == 32
        layer_specs = (
            list(spec.kv_cache_specs.values())
            if hasattr(spec, "kv_cache_specs")
            else [spec] * 32
        )
        assert len(layer_specs) == 32 and all(
            s.dtype == torch.float16 for s in layer_specs
        )
        actual_block_size = spec.block_size
        assert (
            sum(s.page_size_bytes for s in layer_specs)
            == actual_block_size * KV_BYTES_PER_TOKEN
        )
        dtype_counts = Counter()
        for weight in runner.model.parameters():
            dtype_counts[str(weight.dtype)] += weight.numel()
        checkpoint_counts = dict(
            sum(
                (
                    Counter(checkpoint_dtypes(p))
                    for p in sorted(model_dir.glob("*.safetensors"))
                ),
                Counter(),
            )
        )
        assert checkpoint_counts.get("F16", 0) > 6_500_000_000
        assert dtype_counts.get("torch.float16", 0) > 6_500_000_000
        assert config.model_config.quantization is None
        quant_methods = Counter(
            type(m.quant_method).__name__
            for m in runner.model.modules()
            if hasattr(m, "quant_method")
        )
        assert quant_methods["UnquantizedLinearMethod"] >= 32
        metadata = {
            "loaded_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": torch.cuda.get_device_capability(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "before_gpu": before_gpu,
            "after_load_gpu": gpu_snapshot(),
            "geometry": geometry,
            "bytes_per_kv_token": KV_BYTES_PER_TOKEN,
            "parameter_numel_by_dtype": dict(dtype_counts),
            "checkpoint_numel_by_dtype": checkpoint_counts,
            "quant_method_classes": dict(quant_methods),
            "quantization": config.model_config.quantization,
            "model_class": type(runner.model).__name__,
            "tokenizer_class": type(engine.get_tokenizer()).__name__,
            "hf_config": config.model_config.hf_config.to_dict(),
            "cache_config": dataclasses.asdict(config.cache_config),
            "scheduler_config": dataclasses.asdict(config.scheduler_config),
            "compilation_config": TypeAdapter(CompilationConfig).dump_python(
                config.compilation_config, exclude={"static_forward_context": True}
            ),
            "attention_config": {
                "backend_selection": os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
            },
            "attention_backend_classes": sorted(
                {
                    str(type(m.attn.impl))
                    for m in runner.model.modules()
                    if hasattr(m, "attn") and hasattr(m.attn, "impl")
                }
            ),
            "kv_cache_config": dataclasses.asdict(kv_config),
            "kv_block_size": actual_block_size,
            "kv_pool_blocks_including_null": kv_config.num_blocks,
            "kv_pool_token_slots_including_null": kv_config.num_blocks
            * actual_block_size,
            "kv_pool_bytes_from_geometry": kv_config.num_blocks
            * actual_block_size
            * KV_BYTES_PER_TOKEN,
            "runner_model_memory_usage": getattr(runner, "model_memory_usage", None),
            "cuda_allocated_after_load": torch.cuda.memory_allocated(),
            "cuda_reserved_after_load": torch.cuda.memory_reserved(),
            "method": "synchronized offline in-process LLMEngine.step; current-stream execute_model events; no HTTP; prefill eager; decode CUDA graphs, no torch.compile",
        }
        write_json(output_dir / "loaded_metadata.json", metadata)
        print("LOAD_DONE", metadata["kv_pool_token_slots_including_null"], flush=True)

        records = []
        event_pair = None
        schedule_original = scheduler.schedule
        execute_original = core.model_executor.execute_model

        def schedule_hook(*a, **kw):
            previous = {k: r.num_computed_tokens for k, r in scheduler.requests.items()}
            output = schedule_original(*a, **kw)
            preempted = [
                k
                for k, v in previous.items()
                if k in scheduler.requests
                and scheduler.requests[k].num_computed_tokens < v
            ]
            scheduled = []
            for request_id, count in output.num_scheduled_tokens.items():
                request = scheduler.requests[request_id]
                computed_before = request.num_computed_tokens - count
                prompt_work, decode_work = classify(
                    computed_before, request.num_prompt_tokens, count
                )
                block_counts = [
                    len(blocks) for blocks in manager.get_block_ids(request_id)
                ]
                scheduled.append(
                    {
                        "request_id": request_id,
                        "scheduled_tokens": count,
                        "computed_before": computed_before,
                        "prompt_tokens": request.num_prompt_tokens,
                        "prompt_work": prompt_work,
                        "decode_work": decode_work,
                        "allocated_blocks": block_counts,
                    }
                )
            records.append(
                {
                    "scheduled": scheduled,
                    "scheduled_total_tokens": output.total_num_scheduled_tokens,
                    "preempted_request_ids": preempted,
                    "kv_usage_after_schedule": manager.usage,
                }
            )
            return output

        def execute_hook(*a, **kw):
            if event_pair is not None:
                event_pair[0].record()
            result = execute_original(*a, **kw)
            if event_pair is not None:
                event_pair[1].record()
            return result

        scheduler.schedule = schedule_hook
        core.model_executor.execute_model = execute_hook
        run_serial = 0
        deadline = time.monotonic() + args.minutes * 60

        def run_cell(
            length, batch, repeat, kind, output_len=OUTPUT_LEN, instrument=True
        ):
            nonlocal event_pair, run_serial
            assert not engine.has_unfinished_requests()
            run_serial += 1
            chunk = TOKEN_BUDGET // batch
            scheduler.scheduler_config.long_prefill_token_threshold = chunk
            rng = random.Random(20261103 + run_serial)
            prompts = [
                [rng.randrange(1000, 30000) for _ in range(length)]
                for _ in range(batch)
            ]
            params = SamplingParams(
                temperature=0, max_tokens=output_len, ignore_eos=True, detokenize=False
            )
            ids = [
                f"{kind}-{length}-{batch}-{repeat}-{run_serial}-{i}"
                for i in range(batch)
            ]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            for request_id, prompt in zip(ids, prompts):
                engine.add_request(request_id, {"prompt_token_ids": prompt}, params)
            enqueue_ms = (time.perf_counter() - begun) * 1000
            steps, finals = [], {}
            while engine.has_unfinished_requests():
                if thermal_stop.is_set():
                    append_json(
                        output_dir / "partial_runs.jsonl",
                        {
                            "status": "thermal_abort",
                            "kind": kind,
                            "input_tokens": length,
                            "requested_batch": batch,
                            "repeat": repeat,
                            "steps": steps,
                            "final_outputs": finals,
                            "thermal_state": dict(thermal_state),
                        },
                    )
                check_cooling()
                assert time.monotonic() < deadline, "Compute deadline reached"
                records.clear()
                event_pair = (
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    if instrument
                    else None
                )
                if event_pair is not None:
                    event_pair[0].record()
                    event_pair[1].record()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = engine.step()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000
                assert len(records) == 1, "One schedule per step is required"
                row = records[0]
                row["engine_step_wall_ms"] = wall_ms
                row["forward_executor_cuda_span_ms"] = (
                    event_pair[0].elapsed_time(event_pair[1])
                    if event_pair is not None
                    else None
                )
                p = sum(r["prompt_work"] for r in row["scheduled"])
                d = sum(r["decode_work"] for r in row["scheduled"])
                row["phase"] = (
                    "mixed"
                    if p and d
                    else "prefill" if p else "decode" if d else "empty"
                )
                row["active_batch"] = len(row["scheduled"])
                row["outputs"] = [
                    {
                        "request_id": out.request_id,
                        "output_tokens": len(out.outputs[0].token_ids),
                        "finished": out.finished,
                        "finish_reason": out.outputs[0].finish_reason,
                    }
                    for out in outputs
                ]
                for out in outputs:
                    if out.finished:
                        finals[out.request_id] = {
                            "output_tokens": len(out.outputs[0].token_ids),
                            "finish_reason": out.outputs[0].finish_reason,
                        }
                steps.append(row)
            elapsed_ms = (time.perf_counter() - begun) * 1000
            event_pair = None
            assert len(finals) == batch and all(
                o["output_tokens"] == output_len for o in finals.values()
            )
            assert all(
                not s["preempted_request_ids"] for s in steps
            ), "Preemption invalidates homogeneous cell"
            assert all(
                s["active_batch"] == batch and s["phase"] != "mixed" for s in steps
            )
            prefill = [s for s in steps if s["phase"] == "prefill"]
            decode = [s for s in steps if s["phase"] == "decode"]
            assert len(prefill) == math.ceil(length / chunk)
            assert len(decode) == output_len - 1
            assert sum(s["scheduled_total_tokens"] for s in prefill) == length * batch
            assert all(s["scheduled_total_tokens"] == batch for s in decode)
            occupied = max(
                sum(r["allocated_blocks"][0] for r in s["scheduled"]) for s in steps
            )
            expected_blocks = batch * math.ceil(
                (length + output_len - 1) / actual_block_size
            )
            assert occupied == expected_blocks, (occupied, expected_blocks)
            result = {
                "status": "ok",
                "kind": kind,
                "input_tokens": length,
                "requested_batch": batch,
                "repeat": repeat,
                "run_serial": run_serial,
                "output_tokens_per_request": output_len,
                "prefill_chunk_limit_per_request": chunk,
                "instrumented": instrument,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompts).encode()
                ).hexdigest(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "enqueue_ms": enqueue_ms,
                "end_to_end_ms": elapsed_ms,
                "prefill_engine_ms": sum(s["engine_step_wall_ms"] for s in prefill),
                "prefill_forward_executor_cuda_ms": (
                    sum(s["forward_executor_cuda_span_ms"] for s in prefill)
                    if instrument
                    else None
                ),
                "decode_engine_step_mean_ms": (
                    statistics.mean(s["engine_step_wall_ms"] for s in decode)
                    if decode
                    else None
                ),
                "decode_forward_executor_cuda_mean_ms": (
                    statistics.mean(s["forward_executor_cuda_span_ms"] for s in decode)
                    if decode and instrument
                    else None
                ),
                "decode_engine_step_stdev_ms": (
                    statistics.stdev(s["engine_step_wall_ms"] for s in decode)
                    if len(decode) > 1
                    else None
                ),
                "peak_occupied_request_blocks": occupied,
                "peak_request_kv_bytes": occupied
                * actual_block_size
                * KV_BYTES_PER_TOKEN,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "final_outputs": finals,
                "steps": steps,
            }
            append_json(output_dir / "runs.jsonl", result)
            print(
                "CELL",
                kind,
                length,
                batch,
                repeat,
                round(result["prefill_engine_ms"], 3),
                result["decode_engine_step_mean_ms"],
                flush=True,
            )
            return result

        smoke_params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=False)
        engine.add_request(
            "text-smoke", "What is 2 plus 2? Answer briefly.", smoke_params
        )
        smoke = []
        while engine.has_unfinished_requests():
            check_cooling()
            for out in engine.step():
                if out.finished:
                    smoke.append(
                        {
                            "text": out.outputs[0].text,
                            "tokens": len(out.outputs[0].token_ids),
                            "finish_reason": out.outputs[0].finish_reason,
                        }
                    )
        assert smoke and smoke[0]["tokens"] > 0
        write_json(output_dir / "smoke.json", smoke)
        screen_start = time.monotonic()
        while time.monotonic() - screen_start < 15 and screen_count < 12:
            run_cell(1024, 8, screen_count, "thermal_screen")
            screen_count += 1
        check_cooling()
        assert thermal_state["samples"] > 0, "No cooling telemetry received"
        write_json(
            output_dir / "cooling_screen.json",
            {
                "status": "passed",
                "elapsed_s": time.monotonic() - screen_start,
                "runs": screen_count,
                "thermal_state": dict(thermal_state),
            },
        )
        print(
            "COOLING_SCREEN_PASSED", screen_count, thermal_state["latest"], flush=True
        )
        run_serial = 0
        run_cell(128, 1, 0, "phase_check_one_output", output_len=1)
        for length in ([128, 1024] if args.quick else [128, 512, 1024, 1920]):
            for batch in ([1, 4] if args.quick else [1, 2, 4, 8, 16]):
                required_blocks = batch * math.ceil(
                    (length + OUTPUT_LEN - 1) / actual_block_size
                )
                if required_blocks > kv_config.num_blocks - 1:
                    append_json(
                        output_dir / "failures.jsonl",
                        {
                            "input_tokens": length,
                            "batch": batch,
                            "status": "capacity_exceeded",
                            "reason": "Required FP16 KV blocks exceed the configured pool; not executed",
                            "required_blocks": required_blocks,
                            "usable_pool_blocks": kv_config.num_blocks - 1,
                        },
                    )
                    print(
                        "SKIP_CAPACITY",
                        length,
                        batch,
                        required_blocks,
                        kv_config.num_blocks - 1,
                        flush=True,
                    )
                    continue
                try:
                    run_cell(length, batch, -1, "warmup")
                    for repeat in range(args.repeats):
                        run_cell(length, batch, repeat, "measured")
                except torch.OutOfMemoryError:
                    append_json(
                        output_dir / "failures.jsonl",
                        {
                            "input_tokens": length,
                            "batch": batch,
                            "status": "OOM",
                            "utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise
        for repeat in range(3):
            run_cell(1024, 4, repeat, "events_off_check", instrument=False)
            run_cell(1024, 4, repeat, "events_on_check")
        write_json(
            output_dir / "completion.json",
            {
                "status": "complete",
                "utc": datetime.now(timezone.utc).isoformat(),
                "after_gpu": gpu_snapshot(),
                "run_count": run_serial,
                "thermal_screen_runs": screen_count,
                "total_runs": run_serial + screen_count,
                "thermal_state": dict(thermal_state),
            },
        )
    except ThermalStop as error:
        write_json(
            output_dir / "completion.json",
            {
                "status": "thermal_abort",
                "utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(error),
                "thermal_state": dict(thermal_state),
            },
        )
        print("THERMAL_ABORT", str(error), flush=True)
        return 2
    finally:
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        finally:
            shutting_down.set()
            telemetry.terminate()
            telemetry.wait(timeout=15)
            monitor.join(timeout=5)
            telemetry_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile LLaMA-7B FP16 with vLLM 0.11.0.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minutes", type=float, default=25)
    arguments = parser.parse_args()
    if arguments.gpu < 0 or arguments.repeats < 3 or not 0 < arguments.minutes <= 30:
        parser.error("Require gpu >= 0, repeats >= 3 and 0 < minutes <= 30")
    self_check()
    if arguments.self_check:
        print("SELF_CHECK_OK")
    else:
        if arguments.model is None or arguments.output is None:
            parser.error("--model and --output are required")
        raise SystemExit(main(arguments))
