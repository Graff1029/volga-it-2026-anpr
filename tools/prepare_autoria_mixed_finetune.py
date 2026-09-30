"""Подготовить один воспроизводимый mixed OCR experiment: AUTO.RIA crops + synthetic.

Источник AUTO.RIA содержит только тесные OCR-кропы standard type1. Он не
становится набором vehicle scenes и не добавляется в конкурсный dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_finetune_common import MODEL_HEIGHT, batch_tensors, load_jsonl, save_json

ALPHABET = "ABEKMHOPCTYX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-train", type=Path, required=True,
                        help="Распакованный train.zip AUTO.RIA.")
    parser.add_argument("--real-validation", type=Path, required=True,
                        help="Распакованный val.zip AUTO.RIA.")
    parser.add_argument("--synthetic", type=Path,
                        default=Path("dataset/training/ocr_synthetic_official_v1_20260929"))
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/training/ocr_autoria_mixed_20260930"))
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--real-train-count", type=int, default=5000)
    parser.add_argument("--real-validation-count", type=int, default=500)
    return parser.parse_args()


def valid_plate(value: str) -> bool:
    if len(value) not in (8, 9):
        return False
    return (value[0] in ALPHABET and value[1:4].isdigit() and
            value[4] in ALPHABET and value[5] in ALPHABET and
            value[6:].isdigit())


def source_images(root: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    result = []
    for path in paths:
        label = path.stem.upper()
        if not valid_plate(label):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"Unreadable AUTO.RIA image: {path}")
        result.append({"path": path, "plate_num": label, "shape": list(image.shape)})
    return result


def known_plates() -> set[str]:
    result: set[str] = set()
    with (ROOT / "evaluation/official_debug/extracted/Полуфинал - public/debug_labels.csv").open(
            encoding="utf-8", newline="") as stream:
        result.update(row["plate_num"] for row in csv.DictReader(stream, delimiter=";")
                      if "#" not in row["plate_num"])
    for path in (ROOT / "verification/update_03/straight_reference.json",
                 ROOT / "verification/update_03/two_line_reference.json"):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("plate_num"):
                result.add(row["plate_num"])
    with (ROOT / "dataset/candidates/official_v1/meta.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            if row["is_synthetic"] == "0":
                result.add(row["plate_num"])
    return result


def unique_records(records: list[dict], excluded: set[str]) -> tuple[list[dict], dict]:
    result, seen, dropped = [], set(), Counter()
    for record in records:
        plate = record["plate_num"]
        if plate in excluded:
            dropped["known_evaluation_plate"] += 1
        elif plate in seen:
            dropped["duplicate_plate_in_source"] += 1
        else:
            result.append(record)
            seen.add(plate)
    return result, dict(dropped)


def choose(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Need {count} unique AUTO.RIA records, only {len(records)} available")
    ordered = sorted(records, key=lambda item: (item["plate_num"], item["path"].as_posix()))
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_real_fragment(root: Path, split: str, index: int, record: dict, rows: list[dict]) -> None:
    image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable selected AUTO.RIA image: {record['path']}")
    name = f"autoria_{index:05d}_{record['plate_num']}.png"
    target = root / "fragments" / split / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(target)
    rows.append({
        "fragment": target.relative_to(root).as_posix(), "split": split,
        "source_kind": "autoria_real_crop", "source_image": record["path"].as_posix(),
        "plate_num": record["plate_num"], "plate_type": "type1", "field": "whole_line",
        "target_text": record["plate_num"], "order": 0,
        "pipeline_path": "autoria_full_plate_crop",
    })


def add_synthetic_fragment(root: Path, split: str, index: int, row: dict, rows: list[dict], source: Path) -> None:
    destination = root / "fragments" / split / f"synthetic_{index:05d}_{Path(row['fragment']).name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / row["fragment"], destination)
    rows.append({**row, "fragment": destination.relative_to(root).as_posix(),
                 "split": split, "source_kind": "official_v1_synthetic"})


def select_synthetic_train(rows: list[dict], count: int, seed: int) -> list[dict]:
    types = ("type1", "type1a", "type1b")
    quotas = {kind: count // len(types) for kind in types}
    for kind in types[:count % len(types)]:
        quotas[kind] += 1
    capacities = {kind: sum(row["split"] == "train" and row["plate_type"] == kind
                            for row in rows) for kind in types}
    # Existing official_v1 preparation has fewer type1 fragments than the
    # nominal one-third quota.  Keep every type in the mixed experiment, but
    # redistribute only that unmet part across the other two types instead of
    # silently reducing the synthetic half below the requested 5,000 rows.
    shortfall = 0
    for kind in types:
        if capacities[kind] < quotas[kind]:
            shortfall += quotas[kind] - capacities[kind]
            quotas[kind] = capacities[kind]
    for kind in sorted(types, key=lambda item: (capacities[item] - quotas[item], item), reverse=True):
        added = min(shortfall, capacities[kind] - quotas[kind])
        quotas[kind] += added
        shortfall -= added
        if not shortfall:
            break
    if shortfall:
        raise ValueError(f"Not enough synthetic train fragments for {count}; missing {shortfall}")
    selected = []
    for offset, kind in enumerate(types):
        candidates = [row for row in rows if row["split"] == "train" and row["plate_type"] == kind]
        random.Random(seed + offset).shuffle(candidates)
        if len(candidates) < quotas[kind]:
            raise ValueError(f"Synthetic train fragments {kind}: {len(candidates)} < {quotas[kind]}")
        selected.extend(candidates[:quotas[kind]])
    return selected


def check_batch_height(root: Path, rows: list[dict]) -> None:
    paths = [root / row["fragment"] for row in rows[:12]]
    batch = batch_tensors(paths)
    if batch.shape[2] != MODEL_HEIGHT:
        raise AssertionError(f"Wrong OCR batch shape: {tuple(batch.shape)}")


def write_contact_sheet(root: Path, selected: list[dict]) -> Path:
    cells = []
    for record in selected[:16]:
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (320, 80), interpolation=cv2.INTER_AREA)
        canvas = np.full((110, 320, 3), 255, dtype=np.uint8)
        canvas[:80] = image
        cv2.putText(canvas, record["plate_num"], (8, 103), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(canvas)
    while len(cells) % 4:
        cells.append(np.full((110, 320, 3), 255, dtype=np.uint8))
    sheet = np.vstack([np.hstack(cells[index:index + 4]) for index in range(0, len(cells), 4)])
    target = root / "autoria_visual_filename_label_sample.png"
    if not cv2.imwrite(str(target), sheet):
        raise OSError(target)
    return target


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.real_train_count < 1 or args.real_validation_count < 1:
        raise ValueError("Counts must be positive")
    excluded = known_plates()
    train_candidates, train_dropped = unique_records(source_images(args.real_train), excluded)
    validation_candidates, validation_dropped = unique_records(source_images(args.real_validation), excluded)
    real_train = choose(train_candidates, args.real_train_count, args.seed)
    used = {row["plate_num"] for row in real_train}
    real_validation = choose([row for row in validation_candidates if row["plate_num"] not in used],
                             args.real_validation_count, args.seed + 1)
    if used & {row["plate_num"] for row in real_validation}:
        raise AssertionError("AUTO.RIA train/validation plate overlap")
    if len({sha256(row["path"]) for row in real_train + real_validation}) != len(real_train) + len(real_validation):
        raise AssertionError("AUTO.RIA selected byte duplicates")

    source_synthetic = args.synthetic.resolve()
    synthetic_rows = load_jsonl(source_synthetic / "fragments.jsonl")
    synthetic_train = select_synthetic_train(synthetic_rows, len(real_train), args.seed)
    synthetic_validation = [row for row in synthetic_rows if row["split"] == "validation"]
    if not synthetic_validation:
        raise ValueError("No synthetic validation fragments")

    args.output.mkdir(parents=True)
    train_rows, validation_rows = [], []
    for index, record in enumerate(real_train, 1):
        add_real_fragment(args.output, "train", index, record, train_rows)
    for index, row in enumerate(synthetic_train, 1):
        add_synthetic_fragment(args.output, "train", index, row, train_rows, source_synthetic)
    for index, record in enumerate(real_validation, 1):
        add_real_fragment(args.output, "validation", index, record, validation_rows)
    for index, row in enumerate(synthetic_validation, 1):
        add_synthetic_fragment(args.output, "validation", index, row, validation_rows, source_synthetic)
    random.Random(args.seed).shuffle(train_rows)
    check_batch_height(args.output, train_rows)
    check_batch_height(args.output, validation_rows)
    sheet = write_contact_sheet(args.output, real_validation)

    manifest = {
        "format": "volga_easyocr_mixed_autoria_v1",
        "purpose": "one_fixed_real_crop_plus_synthetic_finetune_not_contest_dataset",
        "seed": args.seed,
        "sources": {
            "real": "AY000554/Car_plate_OCR_dataset, AUTO.RIA, CC BY 4.0; tight OCR crops only",
            "synthetic": args.synthetic.as_posix(),
        },
        "source_counts": {"autoria_train_selected": len(real_train),
                          "autoria_validation_selected": len(real_validation),
                          "synthetic_train_fragments": len(synthetic_train),
                          "synthetic_validation_fragments": len(synthetic_validation)},
        "train_fragment_mix": dict(Counter(row["source_kind"] for row in train_rows)),
        "synthetic_train_types": dict(Counter(row["plate_type"] for row in synthetic_train)),
        "validation_fragment_sources": dict(Counter(row["source_kind"] for row in validation_rows)),
        "excluded_known_evaluation_plates": len(excluded),
        "real_train_drop_reasons": train_dropped,
        "real_validation_drop_reasons": validation_dropped,
        "no_real_train_validation_plate_overlap": True,
        "no_selected_real_byte_duplicates": True,
        "ocr_batch_height": MODEL_HEIGHT,
        "visual_filename_label_sample": sheet.name,
        "notes": [
            "AUTO.RIA images are retained separately under dataset/real_ocr and are OCR crops, not automobile scenes.",
            "Target text for AUTO.RIA is source-declared filename text; it is not human verification.",
            "Official debug, input_images, checks/two_line and real official_v1 records are excluded by known plate text before selection.",
        ],
    }
    (args.output / "fragments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows + validation_rows),
        encoding="utf-8")
    save_json(args.output / "manifest.json", manifest)
    save_json(args.output / "autoria_selection.json", {"train": [{**row, "path": row["path"].as_posix()} for row in real_train],
                                                         "validation": [{**row, "path": row["path"].as_posix()} for row in real_validation]})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
