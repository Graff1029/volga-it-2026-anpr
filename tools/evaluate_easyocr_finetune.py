"""Сравнить базовый и явно указанный checkpoint EasyOCR по exact/CER.

Synthetic validation и real development намеренно считаются раздельно.
Реальный manifest передаётся только после человеческой проверки разметки.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from ocr_finetune_common import cer, create_reader, load_checkpoint, load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/training/ocr_synthetic_20260924"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--checkpoint", type=Path, help="Без параметра оценивается исходный english_g2.pth.")
    parser.add_argument("--real-manifest", type=Path,
                        help="Только отдельный проверенный JSONL; не intake и не предварительная разметка.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-kind",
                        help="Оценить только rows с этим внутренним source_kind (например autoria_real_crop).")
    parser.add_argument("--error-samples", type=int, default=10,
                        help="Сохранить до N первых несовпадений для диагностики.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_text(reader, path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    result = reader.recognize(image, detail=1, paragraph=False, decoder="greedy", batch_size=1,
                              workers=0, contrast_ths=0.0,
                              allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(item[1] for item in result)


def score(root: Path, rows: list[dict], reader, limit: int | None, error_samples: int) -> dict:
    if limit is not None:
        rows = rows[:limit]
    exact, edits, reference_length = 0, 0, 0
    groups: dict[str, list[dict]] = defaultdict(list)
    examples, errors = [], []
    for row in rows:
        predicted = read_text(reader, root / row["fragment"])
        expected = row["target_text"]
        exact += predicted == expected
        edits += cer(expected, predicted)
        reference_length += len(expected)
        groups[row["source_image"]].append({**row, "prediction": predicted})
        if len(examples) < 10:
            examples.append({"fragment": row["fragment"], "expected": expected, "prediction": predicted})
        if predicted != expected and len(errors) < error_samples:
            errors.append({"fragment": row["fragment"], "source_image": row["source_image"],
                           "plate_num": row["plate_num"], "plate_type": row["plate_type"],
                           "field": row["field"], "pipeline_path": row["pipeline_path"],
                           "expected": expected, "prediction": predicted})
    plate_exact, plate_edits, plate_length = 0, 0, 0
    plate_results = []
    for group in groups.values():
        group.sort(key=lambda item: item["order"])
        predicted = "".join(item["prediction"] for item in group)
        expected = group[0]["plate_num"]
        plate_exact += predicted == expected
        plate_edits += cer(expected, predicted)
        plate_length += len(expected)
        plate_results.append({"plate_type": group[0]["plate_type"], "expected": expected,
                              "prediction": predicted})
    by_plate_type = {}
    for plate_type in sorted({item["plate_type"] for item in plate_results}):
        selected = [item for item in plate_results if item["plate_type"] == plate_type]
        edits_by_type = sum(cer(item["expected"], item["prediction"]) for item in selected)
        length_by_type = sum(len(item["expected"]) for item in selected)
        by_plate_type[plate_type] = {
            "plates": len(selected),
            "plate_exact_match": sum(item["expected"] == item["prediction"] for item in selected) / len(selected),
            "plate_cer": edits_by_type / length_by_type if length_by_type else None,
        }
    return {
        "fragments": len(rows),
        "fragment_exact_match": exact / len(rows) if rows else None,
        "fragment_cer": edits / reference_length if reference_length else None,
        "plates": len(groups),
        "plate_exact_match": plate_exact / len(groups) if groups else None,
        "plate_cer": plate_edits / plate_length if plate_length else None,
        "by_plate_type": by_plate_type,
        "examples": examples,
        "error_samples": errors,
    }


def main() -> None:
    args = parse_args()
    reader = create_reader(args.models, args.device)
    weights = "base_english_g2"
    if args.checkpoint:
        load_checkpoint(reader, args.checkpoint, args.device)
        weights = str(args.checkpoint.as_posix())
    rows = [row for row in load_jsonl(args.data / "fragments.jsonl") if row["split"] == "validation"]
    if args.source_kind:
        rows = [row for row in rows if row.get("source_kind") == args.source_kind]
        if not rows:
            raise ValueError(f"Нет validation rows с source_kind={args.source_kind!r}")
    result = {
        "weights": weights,
        "source_kind_filter": args.source_kind,
        "synthetic_validation": score(args.data, rows, reader, args.limit, args.error_samples),
        "real_development": {"status": "not_run_no_human_verified_manifest"},
        "warning": "Synthetic и real development метрики не смешиваются.",
    }
    if args.real_manifest:
        real_rows = load_jsonl(args.real_manifest)
        if any(row.get("annotation_status") != "human_verified" for row in real_rows):
            raise ValueError("real manifest должен содержать только annotation_status=human_verified")
        result["real_development"] = score(args.real_manifest.parent, real_rows, reader,
                                            args.limit, args.error_samples)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
