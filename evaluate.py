"""Локальное точное совпадение. Это НЕ официальный оценщик Волга IT."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

TARGETS = ("type1", "type1a", "type1b")


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Неправильный заголовок CSV: {path}")
        rows = list(reader)
    if any(None in row or any(row.get(key) is None for key in required) for row in rows):
        raise ValueError(f"Неправильное число полей: {path}")
    return rows


def score_rows(truth, predictions):
    def key(row):
        return row["image"], row["plate_num"], row["plate_type"]
    targets = Counter(key(row) for row in truth
                      if row["plate_type"] in TARGETS and row.get("is_vehicle", "1") == "1")
    predicted = Counter(key(row) for row in predictions if row["plate_type"] in TARGETS)
    matched = targets & predicted
    report = {"metric": "local_exact_full_number_and_type", "hash_policy": "literal_not_wildcard"}
    for name in (*TARGETS, "all"):
        def count(counter):
            return sum(n for k, n in counter.items() if name == "all" or k[2] == name)
        tp, actual, proposed = count(matched), count(targets), count(predicted)
        report[name] = {"correct": tp, "truth": actual, "predictions": proposed,
                        "missed_or_wrong": actual - tp, "extra_or_wrong": proposed - tp,
                        "precision": tp / proposed if proposed else None,
                        "recall": tp / actual if actual else None}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default="results/evaluation.json", type=Path)
    args = parser.parse_args()
    truth = read_csv(args.truth, ["image", "plate_num", "plate_type", "is_vehicle"])
    predictions = read_csv(args.predictions, ["image", "plate_num", "plate_type", "confidence"])
    report = score_rows(truth, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
