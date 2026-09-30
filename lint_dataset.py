"""Предварительная проверка датасета. Не заменяет validate_dataset.py жюри."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from plate_text import validate_plate_literal

FIELDS = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle",
          "is_synthetic", "source", "license", "conditions"]


def numbers(value, count):
    # Внутренняя договорённость стартового проекта: числа через запятую.
    result = [float(part.strip()) for part in value.split(",")]
    if len(result) != count or not all(math.isfinite(x) for x in result):
        raise ValueError(f"Ожидается {count} конечных чисел через запятую")
    return result


def inspect_dataset(root, format_mode="strict"):
    from PIL import Image

    root = root.resolve()
    errors, warnings, counts = [], [], Counter()
    unique = defaultdict(set)
    seen_images, hashes = set(), {}
    meta = root / "meta.csv"
    if not meta.is_file():
        return {"errors": ["Нет meta.csv"], "warnings": [], "rows": 0}
    with meta.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != FIELDS:
            errors.append("Ожидаемый заголовок: " + ";".join(FIELDS))
            rows = []
        else:
            rows = list(reader)
    for line, row in enumerate(rows, 2):
        try:
            if None in row or any(row[field] is None for field in FIELDS):
                raise ValueError("Неправильное число полей CSV")
            path = (root / row["image"]).resolve()
            if root not in path.parents:
                raise ValueError("Путь изображения выходит за пределы dataset")
            if not path.is_file():
                raise ValueError("Изображение не найдено")
            if row["plate_type"] not in ("type1", "type1a", "type1b", "other"):
                raise ValueError("Неверный plate_type")
            if row["is_vehicle"] not in ("0", "1") or row["is_synthetic"] not in ("0", "1"):
                raise ValueError("Флаги должны быть 0 или 1")
            if row["plate_type"] != "other" and not validate_plate_literal(
                    row["plate_num"], format_mode=format_mode, plate_type=row["plate_type"]):
                raise ValueError("Номер не соответствует формату ТЗ или содержит кириллицу")
            if not row["source"].strip() or not row["license"].strip():
                raise ValueError("Нужно заполнить источник и лицензию")
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            x, y, bw, bh = numbers(row["bbox"], 4)
            if min(x, y) < 0 or min(bw, bh) <= 0 or x + bw > width or y + bh > height:
                raise ValueError("bbox вне изображения или имеет нулевой размер")
            quad = numbers(row["quad"], 8)
            points = list(zip(quad[::2], quad[1::2]))
            if any(not (0 <= px <= width and 0 <= py <= height) for px, py in points):
                raise ValueError("quad выходит за изображение")
            area2 = sum(points[i][0] * points[(i + 1) % 4][1]
                        - points[(i + 1) % 4][0] * points[i][1] for i in range(4))
            if area2 <= 0:
                raise ValueError("quad должен быть ненулевым, по часовой стрелке в координатах изображения")
            if row["is_synthetic"] == "0":
                counts[row["plate_type"]] += 1
                if "#" not in row["plate_num"]:
                    unique[row["plate_type"]].add(row["plate_num"])
            if path not in seen_images:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes:
                    warnings.append(f"Одинаковые байты: {row['image']} и {hashes[digest]}")
                hashes[digest] = row["image"]
                seen_images.add(path)
                if not (root / "labels" / (path.stem + ".txt")).is_file():
                    errors.append(f"Строка {line}: отсутствует labels/{path.stem}.txt")
            if any(tag in row["license"].upper() for tag in ("-NC", "-ND", "-SA")):
                warnings.append(f"Строка {line}: проверить совместимость лицензии с публикацией CC BY 4.0")
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"Строка {line}: {error}")
    if not rows:
        errors.append("Датасет пуст: реальные данные ещё не добавлены")
    warnings.append("Local lint проверяет internal meta и маски; official YOLO labels/quad проверяет только полученный validate_dataset.py организаторов.")
    warnings.append("Эта проверка не подтверждает права на изображения и не является отчётом официального валидатора.")
    return {"validator": "local_precheck_only", "format_mode": format_mode,
            "rows": len(rows), "images": len(seen_images),
            "real_plate_annotations": dict(counts),
            "unique_readable_real_plates": {key: len(value) for key, value in unique.items()},
            "errors": errors, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--output", type=Path, default=Path("results/dataset_precheck.json"))
    parser.add_argument("--format-mode", choices=["strict", "extended"], default="strict",
                        help="strict: official masks; extended: совместимое имя с теми же правилами.")
    args = parser.parse_args()
    result = inspect_dataset(args.dataset, args.format_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
