"""Подготовить synthetic-only OCR fragments из official_v1 без старого type1b.

Белые type1/type1a используют точный split v4. Новые жёлтые type1b делятся
детерминированно по полному номеру; реальные W01/W02/C01, official debug и
предыдущая жёлтая синтетика сюда не попадают.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_finetune_common import save_json
from prepare_ocr_finetune import DEFAULT_EXCLUSIONS, load_excluded, prepare_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/candidates/official_v1"))
    parser.add_argument("--old-split", type=Path,
                        default=Path("dataset/training/ocr_synthetic_20260924_v4_full_detector_crop/split_records.json"))
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/training/ocr_synthetic_official_v1_20260929"))
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--resume", action="store_true",
                        help="Безопасно продолжить только этот output по fragments.jsonl.")
    parser.add_argument("--stop-after", type=int,
                        help="Обработать не более N новых images и сохранить прогресс для --resume.")
    return parser.parse_args()


def bbox_values(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if len(values) != 4:
        raise ValueError(f"Ожидался bbox x,y,w,h, получено {value!r}")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"Некорректный bbox: {value!r}")
    return [x, y, x + width, y + height]


def load_records(dataset: Path) -> list[dict]:
    with (dataset / "meta.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    records = []
    for row in rows:
        if row["is_synthetic"] != "1":
            continue
        record = dict(row)
        record["bbox_values"] = bbox_values(row["bbox"])
        records.append(record)
    if len(records) != 5000:
        raise ValueError(f"Ожидалось 5000 synthetic records official_v1, получено {len(records)}")
    if Counter(record["plate_type"] for record in records) != {"type1": 1667, "type1a": 1667, "type1b": 1666}:
        raise ValueError("Состав synthetic official_v1 не равен 1667/1667/1666")
    if len({record["image"] for record in records}) != len(records):
        raise ValueError("В official_v1 synthetic повторяется image")
    if len({record["plate_num"] for record in records}) != len(records):
        raise ValueError("В official_v1 synthetic повторяется plate_num")
    return records


def select_split(records: list[dict], old_split_path: Path, seed: int) -> tuple[list[dict], list[dict]]:
    by_image = {record["image"]: record for record in records}
    old_split = json.loads(old_split_path.read_text(encoding="utf-8"))
    train, validation = [], []
    for split_name, destination in (("train", train), ("validation", validation)):
        for old in old_split[split_name]:
            if old["plate_type"] not in ("type1", "type1a"):
                continue
            record = by_image.get(old["image"])
            if record is None or record["plate_type"] != old["plate_type"]:
                raise ValueError(f"Не найден белый record старого split: {old['image']}")
            if record["plate_num"] != old["plate_num"]:
                raise ValueError(f"Изменился номер белого record: {old['image']}")
            destination.append(record)
    if Counter(record["plate_type"] for record in train) != {"type1": 1500, "type1a": 1500}:
        raise ValueError("Белый train не совпадает с v4 split")
    if Counter(record["plate_type"] for record in validation) != {"type1": 167, "type1a": 167}:
        raise ValueError("Белая validation не совпадает с v4 split")

    yellow = sorted((record for record in records if record["plate_type"] == "type1b"),
                    key=lambda record: (record["plate_num"], record["image"]))
    rng = random.Random(seed)
    rng.shuffle(yellow)
    validation.extend(yellow[:166])
    train.extend(yellow[166:])
    if len(train) != 4500 or len(validation) != 500:
        raise AssertionError("Ожидались split 4500/500")
    if {record["plate_num"] for record in train} & {record["plate_num"] for record in validation}:
        raise AssertionError("Номер пересекает train и validation")
    return train, validation


def expected_target(fragment: dict) -> str:
    plate, kind, field = fragment["plate_num"], fragment["plate_type"], fragment["field"]
    if kind == "type1a":
        return {"top": plate[:4], "bottom_letters": plate[4:6], "region": plate[6:],
                "top_line": plate[:4], "bottom_line": plate[4:]}[field]
    body_length = 5 if kind == "type1b" else 6
    return {"main": plate[:body_length], "region": plate[body_length:], "whole_line": plate}[field]


def verify_fragments(root: Path, fragments: list[dict]) -> dict:
    mismatches, unreadable, type1b_main_bad = [], [], []
    for fragment in fragments:
        expected = expected_target(fragment)
        if fragment["target_text"] != expected:
            mismatches.append({"fragment": fragment["fragment"], "expected": expected,
                               "actual": fragment["target_text"]})
        if cv2.imread(str(root / fragment["fragment"]), cv2.IMREAD_GRAYSCALE) is None:
            unreadable.append(fragment["fragment"])
        if fragment["plate_type"] == "type1b" and fragment["field"] == "main" and len(fragment["target_text"]) != 5:
            type1b_main_bad.append(fragment["fragment"])
    if mismatches or unreadable or type1b_main_bad:
        raise AssertionError({"target_mismatches": mismatches[:3], "unreadable": unreadable[:3],
                              "type1b_main_bad": type1b_main_bad[:3]})
    return {"fragments_checked": len(fragments), "target_field_mismatches": 0,
            "unreadable_fragments": 0, "type1b_main_fragments": sum(
                item["plate_type"] == "type1b" and item["field"] == "main" for item in fragments),
            "type1b_main_length_violations": 0}


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Уже существует output: {args.output}")
    records = load_records(args.dataset)
    excluded = load_excluded(args.exclusions)
    overlap = sorted({record["image"] for record in records} & excluded)
    if overlap:
        raise ValueError(f"В official_v1 synthetic найден debug file: {overlap[:3]}")
    if args.output.exists():
        split_records = json.loads((args.output / "split_records.json").read_text(encoding="utf-8"))
        train, validation = split_records["train"], split_records["validation"]
        fragments_path = args.output / "fragments.jsonl"
        fragments = ([json.loads(line) for line in fragments_path.read_text(encoding="utf-8").splitlines()]
                     if fragments_path.exists() else [])
    else:
        train, validation = select_split(records, args.old_split, args.seed)
        args.output.mkdir(parents=True)
        save_json(args.output / "split_records.json", {"train": train, "validation": validation})
        fragments = []
    completed = {fragment["source_image"] for fragment in fragments}
    methods: Counter[str] = Counter()
    prior_methods: dict[str, str] = {}
    for fragment in fragments:
        previous = prior_methods.setdefault(fragment["source_image"], fragment["pipeline_path"])
        if previous != fragment["pipeline_path"]:
            raise AssertionError(f"У одного image разные пути фрагментов: {fragment['source_image']}")
    methods.update(prior_methods.values())
    fragments_path = args.output / "fragments.jsonl"
    processed_now = 0
    for split, selected in (("train", train), ("validation", validation)):
        for index, record in enumerate(selected, 1):
            if record["image"] in completed:
                continue
            source = args.dataset / record["image"]
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(source)
            before = len(fragments)
            method = prepare_record(image, args.output, split, record, fragments)
            methods[method] += 1
            with fragments_path.open("a", encoding="utf-8") as stream:
                for fragment in fragments[before:]:
                    stream.write(json.dumps(fragment, ensure_ascii=False) + "\n")
                stream.flush()
            completed.add(record["image"])
            processed_now += 1
            if args.stop_after and processed_now >= args.stop_after:
                print(json.dumps({"status": "partial_progress", "completed_images": len(completed),
                                  "processed_now": processed_now}, ensure_ascii=False))
                return
            if index % 500 == 0:
                print(f"{split}: {index}/{len(selected)}")
    if len(completed) != 5000:
        raise AssertionError(f"Подготовлено {len(completed)} из 5000 image")
    verification = verify_fragments(args.output, fragments)
    manifest = {
        "format": "volga_easyocr_internal_manifest_v2",
        "purpose": "synthetic_only_finetuning_not_official_labels",
        "dataset": args.dataset.as_posix(),
        "old_white_split": args.old_split.as_posix(),
        "seed_new_type1b_split": args.seed,
        "included": {"synthetic": 5000, "real": 0, "old_wrong_mask_type1b": 0,
                     "official_debug": 0},
        "split_counts": {"train_images": len(train), "validation_images": len(validation)},
        "types": {split: dict(sorted(Counter(item["plate_type"] for item in selected).items()))
                  for split, selected in (("train", train), ("validation", validation))},
        "no_plate_num_overlap": True,
        "excluded_debug_file_count": len(excluded),
        "fragment_count": len(fragments), "fragment_methods": dict(sorted(methods.items())),
        "field_target_verification": verification,
        "notes": ["type1/type1a retain exact v4 split records.",
                  "type1b uses official LLDDDRR(R) images only; its main field is [:5].",
                  "Fragments use existing conservative pipeline preparation; no target text selects image boundaries."],
    }
    save_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
