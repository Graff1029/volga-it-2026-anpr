# Данные и воспроизведение обучения

Основной конкурсный dataset, модели и OCR-кропы не публикуются в этом
репозитории. Здесь остаются код генератора и скрипты, необходимые для
воспроизведения методики после самостоятельного получения разрешённых данных.

## Синтетические данные

Генератор `generator/generate_synthetic.py` создаёт type1, type1a и type1b,
сохраняет bbox, quad и метаданные. Источники шрифтов и ресурсов перечислены в
`generator/SOURCES.md`.

## AUTO.RIA OCR-кропы

Источник: https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset

Указанный source commit: `67466c9fc2e6571f190447d812dd3f1cda5589e2`.
В приложенной источником лицензии указаны AUTO.RIA / ARS Online OU
(2018-2024), CC BY 4.0.

Контрольные суммы использованных архивов:

- `train.zip`: `7C53C56F103B7A20316E1E5C995A997CBBC6CADE14B6B156DB0D8269D7E00A60`
- `val.zip`: `70CD48A558ACF3D880FC1827AF97FD721DA7E144BE9B534A7A52CCF6E45D75CC`
- `test.zip`: `1C68D2DCF90AAD2566B231BCC8519C93E886F5A352A996601CBD4F59020FEE1E`

OCR-кропы не являются полными фотографиями автомобиля. Архивы, изображения и
внутренний список отобранных файлов в репозиторий не добавлены.

## Фиксированный mixed fine-tune

Выбранный checkpoint получен от
`easyocr_synth_20260929_official_v1_from_v4_step1000.pt`: 1000 шагов,
batch-size 12, learning rate 1e-5, seed 20260930, CUDA, checkpoint каждые
250 шагов. Train содержал 5000 OCR-кропов AUTO.RIA и 5000 корректных
synthetic fragments. Для подготовки используйте
`tools/prepare_autoria_mixed_finetune.py`, затем
`tools/train_easyocr_finetune.py`.

Official public debug, development-наборы и реальные контрольные сцены не
использовались для обучения mixed fine-tune.
