# Llumnix

Llumnix переносит выполняющиеся LLM-запросы вместе с KV-кэшем между экземплярами модели для балансировки нагрузки, дефрагментации памяти и поддержки приоритетов.

[Задание P24](https://distsys.ru/osukhoroslov/nis2/src/branch/main/project-tasks/p24-llumnix.md) по [статье Llumnix, OSDI 2024](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)

Участники команды № 3:

- Артёмов Михаил
- Фоминых Дарья
- Шевляков Фёдор

Материалы КТ1:

- [Описание проекта](nis/project.md)
- [План эксперимента](nis/experiment-plan.md)
- [Отчёт КТ1](nis/checkpoints/01.md)
- [Ручные примеры политик](nis/checkpoints/01.md#расчётная-проверка)
- [Учебный расчёт метрик и способ повторения](nis/checkpoints/01.md#проверка-расчёта-метрик)
- [Модель и источник профилей](nis/checkpoints/01.md#модель-и-источник-профилей)
