"""Подготовить воспроизводимый train/validation и фрагменты для EasyOCR.

Берутся только 5000 записей завершённой train_candidate-партии.  Все старые
debug-кадры проверяются по явному EXCLUDED_DEBUG_FILES.csv и не могут попасть
ни в split, ни в manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

# При запуске ``python tools/…`` корень проекта не входит в sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_finetune_common import save_json
from plate_regions import split_number_region
from pipeline import text_band
from square_regions import split_square_fields


DEFAULT_SUMMARY = Path("results/train5000_20260923_synthetic.json")
DEFAULT_EXCLUSIONS = Path("dataset/generator/EXCLUDED_DEBUG_FILES.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/training/ocr_synthetic_20260924"))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--reuse-split-records", type=Path,
                        help="Повторно использовать точный ранее проверенный split_records.json.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Явно заменить только ранее созданный output.")
    return parser.parse_args()


def load_excluded(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["image"].replace("\\", "/") for row in csv.DictReader(stream, delimiter=";")}


def allocate_validation(records: list[dict], requested: int) -> dict[str, int]:
    counts = Counter(record["plate_type"] for record in records)
    if requested >= len(records):
        raise ValueError("validation-count должен быть меньше числа изображений")
    raw = {kind: counts[kind] * requested / len(records) for kind in counts}
    result = {kind: int(raw[kind]) for kind in counts}
    for kind, _ in sorted(raw.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True)[:requested - sum(result.values())]:
        result[kind] += 1
    return result


def crop_from_record(image, record: dict):
    x1, y1, x2, y2 = record["bbox_values"]
    height, width = image.shape[:2]
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(-(-x2 // 1))), min(height, int(-(-y2 // 1)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Пустой bbox: {record['image']}")
    return crop


def add_fragment(fragments: list[dict], image, root: Path, split: str, record: dict,
                 field: str, text: str, method: str, order: int) -> None:
    if not text:
        raise ValueError(f"Пустая строка в {record['image']}")
    target = root / "fragments" / split / f"{Path(record['image']).stem}__{field}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Не удалось записать {target}")
    fragments.append({
        "fragment": target.relative_to(root).as_posix(),
        "split": split,
        "source_image": record["image"],
        "plate_num": record["plate_num"],
        "plate_type": record["plate_type"],
        "field": field,
        "target_text": text,
        "order": order,
        "pipeline_path": method,
    })


def prepare_record(image, root: Path, split: str, record: dict, fragments: list[dict]) -> str:
    """Сохранить ровно те полосы, которые в первую очередь использует pipeline.

    Если его эвристика отказывается от разделения, сохраняем его
    консервативный fallback: весь detector crop для строки либо две половины
    полного crop для 1а. Он не выделяет границы по target_text.
    """
    crop = crop_from_record(image, record)
    plate = record["plate_num"]
    if record["plate_type"] == "type1a":
        bands, details = split_square_fields(crop)
        if bands is not None:
            for order, (field, text, band) in enumerate(zip(
                    ("top", "bottom_letters", "region"),
                    (plate[:4], plate[4:6], plate[6:]), bands)):
                padded = cv2.copyMakeBorder(band, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
                add_fragment(fragments, padded, root, split, record, field, text,
                             "square_three_fields", order)
            return "square_three_fields"
        fallback = crop
        split_at = max(1, round(fallback.shape[0] * 0.52))
        add_fragment(fragments, text_band(cv2.cvtColor(fallback[:split_at], cv2.COLOR_BGR2GRAY)),
                     root, split, record, "top_line", plate[:4], "two_lines_fallback", 0)
        add_fragment(fragments, text_band(cv2.cvtColor(fallback[split_at:], cv2.COLOR_BGR2GRAY)),
                     root, split, record, "bottom_line", plate[4:], "two_lines_fallback", 1)
        return f"two_lines_fallback:{details.get('status')}"

    body_length = 5 if record["plate_type"] == "type1b" else 6
    bands, details = split_number_region(crop, body_length)
    if bands is not None:
        for order, (field, text, band) in enumerate(zip(
                ("main", "region"), (plate[:body_length], plate[body_length:]), bands)):
            padded = cv2.copyMakeBorder(band, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            add_fragment(fragments, padded, root, split, record, field, text,
                         "number_and_region", order)
        return "number_and_region"
    fallback = crop
    gray = text_band(cv2.cvtColor(fallback, cv2.COLOR_BGR2GRAY))
    add_fragment(fragments, gray, root, split, record, "whole_line", plate,
                 "whole_line_fallback", 0)
    return f"whole_line_fallback:{details.get('status')}"


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    records = summary.get("records", [])
    excluded = load_excluded(args.exclusions)
    expected = 5000
    if summary.get("status") != "completed" or len(records) != expected:
        raise ValueError("Ожидалась завершённая партия ровно из 5000 кадров")
    if summary.get("dataset_role") != "train_candidate":
        raise ValueError("Summary не помечен train_candidate")
    paths = [record["image"].replace("\\", "/") for record in records]
    plates = [record["plate_num"] for record in records]
    if len(set(paths)) != expected or len(set(plates)) != expected:
        raise ValueError("В 5000-кандидатах повторяется image или plate_num")
    overlap = sorted(set(paths) & excluded)
    if overlap:
        raise ValueError(f"В train_candidate обнаружен debug-файл: {overlap[:3]}")
    if len(excluded) != 132:
        raise ValueError(f"Ожидалось 132 явно исключённых debug-файла, найдено {len(excluded)}")

    split_source = None
    if args.reuse_split_records:
        split_source = args.reuse_split_records
        reused = json.loads(split_source.read_text(encoding="utf-8"))
        train, validation = reused.get("train", []), reused.get("validation", [])
        if {row["image"] for row in train + validation} != set(paths):
            raise ValueError("reuse split не содержит ровно исходные 5000 image")
        if any(row["image"] not in set(paths) for row in train + validation):
            raise ValueError("reuse split содержит image вне исходной партии")
        validation_by_type = Counter(row["plate_type"] for row in validation)
    else:
        validation_by_type = allocate_validation(records, args.validation_count)
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(record["plate_type"], []).append(record)
        rng = random.Random(args.seed)
        validation, train = [], []
        for kind in sorted(grouped):
            bucket = sorted(grouped[kind], key=lambda value: value["image"])
            rng.shuffle(bucket)
            validation.extend(bucket[:validation_by_type[kind]])
            train.extend(bucket[validation_by_type[kind]:])
    if len(train) != 4500 or len(validation) != 500:
        raise AssertionError("Неверный размер split")
    if set(record["plate_num"] for record in train) & set(record["plate_num"] for record in validation):
        raise AssertionError("Номер попал и в train, и в validation")

    output = args.output
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Уже существует {output}; для пересоздания нужен --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    fragments: list[dict] = []
    method_counts: Counter[str] = Counter()
    for split, selected in (("train", train), ("validation", validation)):
        for index, record in enumerate(selected, 1):
            source = args.dataset / record["image"]
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(source)
            method_counts[prepare_record(image, output, split, record, fragments)] += 1
            if index % 500 == 0:
                print(f"{split}: {index}/{len(selected)}")

    manifest = {
        "format": "volga_easyocr_internal_manifest_v1",
        "purpose": "internal_finetuning_not_official_labels",
        "seed": args.seed,
        "source_summary": args.summary.as_posix(),
        "source_operation": summary["operation_id"],
        "source_count": expected,
        "excluded_debug_file_count": len(excluded),
        "split_counts": {"train_images": len(train), "validation_images": len(validation)},
        "types": {
            split: dict(sorted(Counter(record["plate_type"] for record in selected).items()))
            for split, selected in (("train", train), ("validation", validation))
        },
        "validation_by_type": dict(sorted(validation_by_type.items())),
        "reused_exact_split_records": split_source.as_posix() if split_source else None,
        "no_plate_num_overlap": True,
        "fragment_count": len(fragments),
        "fragment_methods": dict(sorted(method_counts.items())),
        "notes": [
            "Fragments reuse current pipeline split_number_region/split_square_fields and conservative full detector-crop fallback branches.",
            "Detector is not trained; bbox from synthetic metadata approximates detector crop only for internal OCR preparation.",
        ],
    }
    (output / "fragments.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in fragments), encoding="utf-8")
    save_json(output / "manifest.json", manifest)
    save_json(output / "split_records.json", {"train": train, "validation": validation})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
