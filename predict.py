"""Каталог JPG/PNG -> конкурсный CSV и отдельные диагностические файлы."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parent
FIELDS = ["image", "plate_num", "plate_type", "confidence"]


def image_files(folder):
    if not folder.is_dir():
        raise ValueError(f"Нет входной папки: {folder}")
    files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".png")),
                   key=lambda p: (p.name.casefold(), p.name))
    if not files:
        raise ValueError("Во входной папке нет JPG/PNG. Вложенные папки не обходятся.")
    return files


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "confidence": f"{row['confidence']:.6f}"})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/result.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ocr-checkpoint", type=Path,
                        help="Экспериментально загрузить внутренний EasyOCR checkpoint; по умолчанию исходный english_g2.pth.")
    parser.add_argument("--format-mode", choices=["strict", "extended"], default="strict",
                        help="strict: official masks type1/type1a LDDDLLRR(R), type1b LLDDDRR(R); extended is a compatible alias.")
    args = parser.parse_args()
    from pipeline import PlatePipeline, read_image

    files = image_files(args.input)
    with args.settings.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    pipeline = PlatePipeline(args.models, settings, args.device, args.ocr_checkpoint, args.format_mode)
    pipeline.synchronize()
    load_ms = (perf_counter() - start) * 1000
    rows, debug, timings = [], [], []
    # Первый кадр не исключается: прогрева в этой начальной проверке нет.
    start_total = perf_counter()
    for index, path in enumerate(files, 1):
        start = perf_counter()
        frame = read_image(path)
        found, details = pipeline.predict(frame)
        pipeline.synchronize()
        elapsed = (perf_counter() - start) * 1000
        timings.append(elapsed)
        rows.extend({"image": path.name, **row} for row in found)
        debug.append({"image": path.name, "elapsed_ms": elapsed, "detections": details})
        print(f"[{index}/{len(files)}] {path.name}: {len(found)} номер(ов), {elapsed:.1f} ms")
    write_results(args.output, rows)
    total_ms = (perf_counter() - start_total) * 1000
    ordered = sorted(timings)
    report = {
        "stage": "baseline_0_3_not_submission_ready", "images": len(files),
        "rows": len(rows), "mean_image_ms": statistics.mean(timings),
        "p95_image_ms": ordered[max(0, __import__('math').ceil(len(ordered) * 0.95) - 1)],
        "load_models_ms": load_ms, "total_with_csv_ms": total_ms,
        "amortized_with_csv_ms": total_ms / len(files),
        "warmup_images_excluded": 0, "includes_image_decode": True,
        "detector_provider": "CPUExecutionProvider",
        "ocr_device": "cuda" if pipeline.gpu else "cpu",
        "ocr_checkpoint": pipeline.ocr_checkpoint or "base_english_g2",
        "python": sys.version, "platform": platform.platform(),
        "torch": pipeline.torch.__version__,
        "gpu": pipeline.torch.cuda.get_device_name(0) if pipeline.gpu else None,
        "settings": settings, "format_mode": args.format_mode,
    }
    args.output.with_suffix(".benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".debug.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {args.output}. Среднее: {report['mean_image_ms']:.1f} ms.")
    print("Это замер текущего ПК, не подтверждение скорости на ПК жюри.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        # Ошибка должна быть видна пользователю и вызывающей программе.
        raise
