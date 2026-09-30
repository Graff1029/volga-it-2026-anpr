"""Deterministic, transactional generator of Russian-style synthetic plates.

This is an internal data-preparation tool.  Its JSON labels are not an
organizer format; use convert_labels.py only when a documented target format is
available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


VERSION = "synthetic_generator_v2"
INTERNAL_SCHEMA = "volga_synthetic_internal_v1"
SYNTHETIC_LICENSE = "CC BY 4.0"
ALLOWED = "ABEKMHOPCTYX"
# Official README/validator accepts any two- or three-digit region.  The
# generator keeps a documented finite sample, including values that disprove
# the former first-digit 1/2/7 heuristic.
REGIONS = ("02", "16", "54", "70", "77", "102", "116", "154", "197", "323", "550", "702", "777", "778", "790", "799")
TYPE_SPECS = {
    "type1": {"size": (1040, 224), "color": (244, 244, 244)},
    "type1a": {"size": (580, 340), "color": (244, 244, 244)},
    "type1b": {"size": (1040, 224), "color": (0, 210, 255)},
}
FIELDS = [
    "image",
    "plate_num",
    "plate_type",
    "bbox",
    "quad",
    "is_vehicle",
    "is_synthetic",
    "source",
    "license",
    "conditions",
]


class SimulatedInterruption(RuntimeError):
    """Only used by the focused transaction-recovery check."""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_numbers(values: list[float]) -> str:
    return ",".join(f"{value:.2f}" for value in values)


def number_text(rng: np.random.Generator, used_numbers: set[str], regions: tuple[str, ...], plate_type: str) -> str:
    """Create a unique plate identity; the later split must be by this value."""
    while True:
        if plate_type == "type1b":
            candidate = (
                f"{rng.choice(list(ALLOWED))}{rng.choice(list(ALLOWED))}"
                f"{rng.integers(0, 1000):03d}{rng.choice(regions)}"
            )
        else:
            candidate = (
                f"{rng.choice(list(ALLOWED))}{rng.integers(0, 1000):03d}"
                f"{rng.choice(list(ALLOWED))}{rng.choice(list(ALLOWED))}"
                f"{rng.choice(regions)}"
            )
        if candidate not in used_numbers:
            used_numbers.add(candidate)
            return candidate


def centered_text(
    image: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = (int(center[0] - width / 2), int(center[1] + height / 2 - baseline / 2))
    cv2.putText(image, text, origin, font, scale, (18, 18, 18), thickness, cv2.LINE_AA)


def plate_text_fields(plate_type: str, plate_num: str) -> dict[str, str]:
    """The exact plate-number substrings that must be passed to rendering."""
    main_pattern = rf"[{ALLOWED}]\d{{3}}[{ALLOWED}]{{2}}"
    type1b_pattern = rf"[{ALLOWED}]{{2}}\d{{3}}"
    expected = type1b_pattern if plate_type == "type1b" else main_pattern
    if not re.fullmatch(expected + r"\d{2,3}", plate_num):
        raise ValueError(f"Unsupported {plate_type} plate_num: {plate_num}")
    if plate_type == "type1a":
        return {
            "top": plate_num[:4],
            "bottom_letters": plate_num[4:6],
            "region": plate_num[6:],
        }
    body_length = 5 if plate_type == "type1b" else 6
    return {"main": plate_num[:body_length], "region": plate_num[body_length:]}


def plate_image(plate_type: str, plate_num: str, draw_calls: list[dict[str, str]]) -> np.ndarray:
    width, height = TYPE_SPECS[plate_type]["size"]
    plate = np.full((height, width, 3), TYPE_SPECS[plate_type]["color"], dtype=np.uint8)
    fields = plate_text_fields(plate_type, plate_num)

    def draw(field: str, text: str, center: tuple[int, int], scale: float, thickness: int) -> None:
        draw_calls.append({"field": field, "text": text})
        centered_text(plate, text, center, scale, thickness)

    cv2.rectangle(plate, (5, 5), (width - 6, height - 6), (22, 22, 22), 6)
    if plate_type == "type1a":
        # GOST R 50577-2018, Appendix A, figures A.3/A.4: a lower-right
        # region block, not a full-height right section. The block contains
        # the region above the country mark and flag; the two letters remain
        # lower-left and the first four characters are above both.
        region_left = int(width * 0.58)
        region_top = int(height * 0.53)
        region_right = width - 25
        region_bottom = height - 25
        cv2.rectangle(plate, (region_left, region_top), (region_right, region_bottom), (22, 22, 22), 5)
        draw("top", fields["top"], (int(width * 0.46), int(height * 0.32)), 4.0, 8)
        draw("bottom_letters", fields["bottom_letters"], (int(width * 0.25), int(height * 0.75)), 4.0, 8)
        draw("region", fields["region"], (int((region_left + region_right) / 2), int(height * 0.68)), 2.45, 5)
        draw_calls.append({"field": "country", "text": "RUS"})
        centered_text(plate, "RUS", (int(width * 0.68), int(height * 0.86)), 0.72, 2)
        flag_x, flag_y, flag_w, flag_h = int(width * 0.76), int(height * 0.81), 48, 24
        cv2.rectangle(plate, (flag_x, flag_y), (flag_x + flag_w, flag_y + flag_h), (20, 20, 20), 1)
        stripe_h = flag_h // 3
        cv2.rectangle(plate, (flag_x + 1, flag_y + 1), (flag_x + flag_w - 1, flag_y + stripe_h), (245, 245, 245), -1)
        cv2.rectangle(plate, (flag_x + 1, flag_y + stripe_h + 1), (flag_x + flag_w - 1, flag_y + 2 * stripe_h), (185, 75, 40), -1)
        cv2.rectangle(plate, (flag_x + 1, flag_y + 2 * stripe_h + 1), (flag_x + flag_w - 1, flag_y + flag_h - 1), (35, 35, 190), -1)
    else:
        # Type 1B has a five-character body (LLDDD), so its separator moves
        # left slightly; no sixth glyph is fabricated before rendering.
        separator_x = int(width * (0.69 if plate_type == "type1b" else 0.765))
        main_x = int(width * (0.34 if plate_type == "type1b" else 0.36))
        region_x = int(width * (0.845 if plate_type == "type1b" else 0.875))
        cv2.rectangle(plate, (separator_x, 12), (separator_x, height - 13), (35, 35, 35), 3)
        draw("main", fields["main"], (main_x, 125), 3.45, 8)
        draw("region", fields["region"], (region_x, 105), 2.45, 5)
        draw_calls.append({"field": "country", "text": "RUS"})
        cv2.putText(plate, "RUS", (int(width * (0.72 if plate_type == "type1b" else 0.79)), 176), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (25, 25, 25), 2, cv2.LINE_AA)
    return plate


def background(rng: np.random.Generator, width: int, height: int, night: bool) -> np.ndarray:
    base = rng.integers(20, 95) if night else rng.integers(85, 175)
    image = np.full((height, width, 3), int(base), dtype=np.uint8)
    for _ in range(35):
        color = int(np.clip(base + rng.normal(0, 32), 0, 255))
        radius = int(rng.integers(12, max(13, min(width, height) // 5)))
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(image, center, radius, (color, color, color), -1, cv2.LINE_AA)
    noise = rng.normal(0, 8 if night else 5, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def target_quad(rng: np.random.Generator, plate_w: int, plate_h: int, canvas_w: int, canvas_h: int) -> np.ndarray:
    scale = float(rng.uniform(0.34, 0.67))
    width = plate_w * scale
    height = plate_h * scale
    center = np.array([
        rng.uniform(width * 0.65, canvas_w - width * 0.65),
        rng.uniform(height * 0.75, canvas_h - height * 0.75),
    ])
    angle = math.radians(float(rng.uniform(-18, 18)))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32)
    corners[:, 0] *= width
    corners[:, 1] *= height
    points = corners @ rotation.T + center
    perspective = rng.normal(0, min(width, height) * 0.055, (4, 2))
    quad = points + perspective
    # Rotation can extend farther than the unrotated centre margin. Translate
    # the completed quad into the frame so labels never describe off-image
    # pixels. Translation preserves the same projective geometry.
    frame_margin = 2.0
    low = quad.min(axis=0)
    high = quad.max(axis=0)
    shift = np.array(
        [
            frame_margin - low[0] if low[0] < frame_margin else (canvas_w - frame_margin - high[0] if high[0] > canvas_w - frame_margin else 0.0),
            frame_margin - low[1] if low[1] < frame_margin else (canvas_h - frame_margin - high[1] if high[1] > canvas_h - frame_margin else 0.0),
        ]
    )
    quad += shift
    return quad.astype(np.float32)


def apply_effects(
    rng: np.random.Generator, image: np.ndarray, quad: np.ndarray, night: bool
) -> tuple[np.ndarray, list[str]]:
    """Effects use only the seeded RNG: independent from class and index."""
    effects: list[str] = ["night" if night else "day", "perspective"]
    output = image.astype(np.float32)
    gain = float(rng.uniform(0.45, 0.82) if night else rng.uniform(0.82, 1.22))
    output *= gain
    if rng.random() < 0.38:
        center = tuple(np.mean(quad, axis=0).astype(int))
        overlay = np.zeros_like(output)
        radius = int(max(24, np.linalg.norm(quad[1] - quad[0]) * rng.uniform(0.18, 0.42)))
        cv2.circle(overlay, center, radius, (255, 245, 215), -1, cv2.LINE_AA)
        output = cv2.addWeighted(output, 1.0, overlay, float(rng.uniform(0.14, 0.36)), 0)
        effects.append("glare")
    if rng.random() < 0.34:
        length = int(rng.integers(3, 8))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        output = cv2.filter2D(output, -1, kernel)
        effects.append("motion_blur")
    if rng.random() < 0.28:
        encoded = cv2.imencode(".jpg", np.clip(output, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(42, 76))])[1]
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)
        effects.append("jpeg_artifact")
    noise = rng.normal(0, float(rng.uniform(1.5, 9.5)), output.shape)
    return np.clip(output + noise, 0, 255).astype(np.uint8), effects


def internal_label(image: str, plate_type: str, plate_num: str, bbox: list[float], quad: list[float], effects: list[str], rendered_text: list[dict[str, str]]) -> dict:
    return {
        "schema": INTERNAL_SCHEMA,
        "image": image,
        "is_synthetic": 1,
        "is_vehicle": 0,
        "plate_type": plate_type,
        "plate_num": plate_num,
        "bbox_xyxy": bbox,
        "quad_xy": quad,
        "render_effects": effects,
        "rendered_text": rendered_text,
    }


def ensure_meta_header(meta_path: Path) -> None:
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter=";"), [])
        if header != FIELDS:
            raise ValueError(f"Unexpected meta header in {meta_path}")
        return
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=";", lineterminator="\n").writerow(FIELDS)
    atomic_write_text(meta_path, buffer.getvalue())


def read_meta(meta_path: Path) -> list[dict[str, str]]:
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def append_meta_atomic(meta_path: Path, rows: list[dict[str, str]]) -> None:
    existing = read_meta(meta_path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(existing)
    writer.writerows(rows)
    atomic_write_text(meta_path, buffer.getvalue())


def remove_meta_rows_atomic(meta_path: Path, records: list[dict]) -> None:
    own_images = {record["image"] for record in records}
    remaining = [row for row in read_meta(meta_path) if row.get("image") not in own_images]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(remaining)
    atomic_write_text(meta_path, buffer.getvalue())


def transaction_path(dataset: Path, operation_id: str) -> Path:
    return dataset / "generator" / "transactions" / f"{operation_id}.json"


def write_transaction(dataset: Path, transaction: dict) -> None:
    atomic_write_text(transaction_path(dataset, transaction["operation_id"]), json.dumps(transaction, ensure_ascii=False, indent=2) + "\n")


def manifest_path(dataset: Path) -> Path:
    return dataset / "generator" / "batches.json"


def update_manifest(dataset: Path, summary: dict) -> None:
    path = manifest_path(dataset)
    batches = read_json(path, [])
    batches = [entry for entry in batches if entry.get("operation_id") != summary["operation_id"]]
    batches.append(summary)
    atomic_write_text(path, json.dumps(batches, ensure_ascii=False, indent=2) + "\n")


def remove_from_manifest(dataset: Path, operation_id: str) -> None:
    path = manifest_path(dataset)
    batches = [entry for entry in read_json(path, []) if entry.get("operation_id") != operation_id]
    atomic_write_text(path, json.dumps(batches, ensure_ascii=False, indent=2) + "\n")


def output_paths(dataset: Path, record: dict) -> tuple[Path, Path]:
    return dataset / record["image"], dataset / "labels" / record["label_file"]


def records_are_committed(dataset: Path, records: list[dict]) -> bool:
    return bool(records) and all(image.is_file() and label.is_file() for image, label in (output_paths(dataset, record) for record in records))


def rollback_own_output(dataset: Path, records: list[dict]) -> None:
    for image, label in (output_paths(dataset, record) for record in records):
        for path in (image, label):
            if path.exists():
                path.unlink()


def recover_transactions(dataset: Path) -> list[str]:
    directory = dataset / "generator" / "transactions"
    if not directory.is_dir():
        return []
    recovered: list[str] = []
    for path in sorted(directory.glob("*.json")):
        transaction = read_json(path, {})
        if transaction.get("generator") != VERSION or transaction.get("status") in {"completed", "rolled_back"}:
            continue
        records = transaction.get("records", [])
        stage = Path(transaction.get("stage_dir", ""))
        status = transaction.get("status")
        if status == "rolling_back":
            remove_meta_rows_atomic(dataset / "meta.csv", records)
            rollback_own_output(dataset, records)
            transaction["status"] = "rolled_back"
            transaction["recovery"] = "requested replacement completed"
        elif status == "meta_written" and records_are_committed(dataset, records) and transaction.get("summary"):
            update_manifest(dataset, transaction["summary"])
            transaction["status"] = "completed"
            transaction["recovery"] = "metadata-written operation completed from its journal"
        else:
            # A process can die after any image move or after meta.csv has been
            # atomically replaced but before the journal advances.  Removing
            # only the rows named in this journal makes that operation wholly
            # absent before its next deterministic run starts.
            remove_meta_rows_atomic(dataset / "meta.csv", records)
            rollback_own_output(dataset, records)
            transaction["status"] = "rolled_back"
            transaction["recovery"] = "incomplete operation rolled back by its own manifest"
        if stage.is_dir() and dataset.resolve() in stage.resolve().parents:
            shutil.rmtree(stage)
        write_transaction(dataset, transaction)
        recovered.append(transaction.get("operation_id", path.stem))
    return recovered


def create_contact_sheet(records: list[dict], dataset: Path, output_path: Path) -> None:
    thumb_w, thumb_h = 320, 190
    rows = max(1, math.ceil(len(records) / 10))
    canvas = np.full((rows * thumb_h, 10 * thumb_w, 3), 238, dtype=np.uint8)
    for position, record in enumerate(records):
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["image"])
        resized = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        quad = np.array(record["quad_values"], dtype=np.float32).reshape(-1, 2)
        quad[:, 0] *= thumb_w / image.shape[1]
        quad[:, 1] *= thumb_h / image.shape[0]
        cv2.polylines(resized, [quad.astype(np.int32)], True, (0, 190, 0), 2, cv2.LINE_AA)
        row, column = divmod(position, 10)
        canvas[row * thumb_h : (row + 1) * thumb_h, column * thumb_w : (column + 1) * thumb_w] = resized
        cv2.putText(canvas, f"{record['plate_type']} {record['plate_num']}", (column * thumb_w + 8, row * thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 20, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def create_large_examples(records: list[dict], dataset: Path, output_dir: Path, all_records: bool = False) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[str] = []
    selected = records if all_records else [next(record for record in records if record["plate_type"] == plate_type) for plate_type in TYPE_SPECS]
    for record in selected:
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = (int(value) for value in record["bbox_values"])
        margin = 30
        crop = image[max(0, y1 - margin) : min(image.shape[0], y2 + margin), max(0, x1 - margin) : min(image.shape[1], x2 + margin)]
        enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        target = output_dir / f"{record['plate_type']}_{record['plate_num']}.png"
        cv2.imwrite(str(target), enlarged)
        examples.append(target.as_posix())
    return examples


def make_records(args: argparse.Namespace, used_numbers: set[str], stage_dir: Path) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    canvas_w, canvas_h = 960, 540
    records: list[dict] = []
    type_positions = Counter()
    for index in range(args.count):
        plate_type = args.plate_types[index % len(args.plate_types)]
        type_position = type_positions[plate_type]
        type_positions[plate_type] += 1
        record_regions = (args.region_cycle[type_position % len(args.region_cycle)],) if args.region_cycle else REGIONS
        plate_num = number_text(rng, used_numbers, record_regions, plate_type)
        draw_calls: list[dict[str, str]] = []
        plate = plate_image(plate_type, plate_num, draw_calls)
        night = bool(rng.integers(0, 2))
        canvas = background(rng, canvas_w, canvas_h, night)
        quad = target_quad(rng, plate.shape[1], plate.shape[0], canvas_w, canvas_h)
        source = np.array([[0, 0], [plate.shape[1] - 1, 0], [plate.shape[1] - 1, plate.shape[0] - 1], [0, plate.shape[0] - 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, quad)
        warped = cv2.warpPerspective(plate, transform, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(np.full(plate.shape[:2], 255, dtype=np.uint8), transform, (canvas_w, canvas_h))
        canvas[mask > 0] = warped[mask > 0]
        rendered, effects = apply_effects(rng, canvas, quad, night)
        bbox_values = [float(quad[:, 0].min()), float(quad[:, 1].min()), float(quad[:, 0].max()), float(quad[:, 1].max())]
        quad_values = [float(value) for point in quad for value in point]
        stem = f"{args.split}_{args.prefix}_{args.seed}_{index:05d}"
        image_rel = Path("images") / "synthetic" / args.split / f"{stem}.png"
        label_file = f"{stem}.txt"
        stage_image = stage_dir / "images" / f"{stem}.png"
        stage_label = stage_dir / "labels" / label_file
        stage_image.parent.mkdir(parents=True, exist_ok=True)
        stage_label.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(stage_image), rendered):
            raise RuntimeError(f"Could not write {stage_image}")
        atomic_write_text(stage_label, json.dumps(internal_label(image_rel.as_posix(), plate_type, plate_num, bbox_values, quad_values, effects, draw_calls), ensure_ascii=False, indent=2) + "\n")
        record = {
            "image": image_rel.as_posix(),
            "label_file": label_file,
            "plate_num": plate_num,
            "plate_type": plate_type,
            "bbox_xyxy": format_numbers(bbox_values),
            "bbox_xywh": format_numbers([bbox_values[0], bbox_values[1], bbox_values[2] - bbox_values[0], bbox_values[3] - bbox_values[1]]),
            "quad_xy": format_numbers(quad_values),
            "bbox_values": bbox_values,
            "quad_values": quad_values,
            "effects": effects,
            "rendered_text": draw_calls,
        }
        records.append(record)
        if args.interrupt_after_staged and len(records) >= args.interrupt_after_staged:
            raise SimulatedInterruption(f"Artificial interruption after {len(records)} staged files")
    return records


def effect_distribution(records: list[dict]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for plate_type in TYPE_SPECS:
        counter = Counter(effect for record in records if record["plate_type"] == plate_type for effect in record["effects"])
        distribution[plate_type] = {name: int(counter[name]) for name in ("day", "night", "perspective", "glare", "motion_blur", "jpeg_artifact")}
    return distribution


def generate(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    meta_path = dataset / "meta.csv"
    ensure_meta_header(meta_path)
    recovered = recover_transactions(dataset)
    operation_id = f"{args.split}_{args.prefix}_{args.seed}_{args.count}"
    journal_path = transaction_path(dataset, operation_id)
    existing_journal = read_json(journal_path, {})
    if existing_journal.get("status") == "completed":
        if not args.replace_completed:
            return existing_journal["summary"] | {"idempotent_rerun": True, "recovered_transactions": recovered}
        transaction = existing_journal
        transaction["status"] = "rolling_back"
        transaction["replacement_requested"] = True
        write_transaction(dataset, transaction)
        remove_meta_rows_atomic(meta_path, transaction["records"])
        rollback_own_output(dataset, transaction["records"])
        transaction["status"] = "rolled_back"
        transaction["recovery"] = "replaced explicitly by the same operation"
        write_transaction(dataset, transaction)
        remove_from_manifest(dataset, operation_id)
        existing_journal = {}
    for batch in read_json(manifest_path(dataset), []):
        if batch.get("seed") == args.seed and batch.get("operation_id") != operation_id:
            raise ValueError(f"Seed {args.seed} is already reserved by {batch.get('operation_id')}; choose a new split seed.")
    existing_rows = read_meta(meta_path)
    used_numbers = {row["plate_num"] for row in existing_rows if row.get("plate_num")}
    stage_dir = dataset / ".synthetic_staging" / operation_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    transaction = {
        "generator": VERSION,
        "operation_id": operation_id,
        "status": "staging",
        "stage_dir": str(stage_dir),
        "records": [],
    }
    write_transaction(dataset, transaction)
    try:
        records = make_records(args, used_numbers, stage_dir)
    except SimulatedInterruption as error:
        transaction["note"] = str(error)
        write_transaction(dataset, transaction)
        raise
    transaction["records"] = records
    transaction["status"] = "committing"
    write_transaction(dataset, transaction)
    for committed, record in enumerate(records, start=1):
        final_image, final_label = output_paths(dataset, record)
        if final_image.exists() or final_label.exists():
            raise FileExistsError(f"Refusing to overwrite existing batch output: {final_image}")
        final_image.parent.mkdir(parents=True, exist_ok=True)
        final_label.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir / "images" / Path(record["image"]).name, final_image)
        os.replace(stage_dir / "labels" / record["label_file"], final_label)
        if args.interrupt_after_committed and committed >= args.interrupt_after_committed:
            raise SimulatedInterruption(f"Artificial interruption after {committed} committed files")
    meta_rows = [
        {
            "image": record["image"], "plate_num": record["plate_num"], "plate_type": record["plate_type"], "bbox": record["bbox_xywh"], "quad": record["quad_xy"],
            "is_vehicle": "0", "is_synthetic": "1", "source": f"{VERSION}:{args.split}:{args.role}", "license": SYNTHETIC_LICENSE, "conditions": ",".join(record["effects"]),
        }
        for record in records
    ]
    append_meta_atomic(meta_path, meta_rows)
    summary = {
        "generator": VERSION, "operation_id": operation_id, "status": "completed", "split": args.split, "prefix": args.prefix, "seed": args.seed, "count": args.count, "dataset_role": args.role,
        "image_root": (dataset / "images" / "synthetic" / args.split).as_posix(), "label_root": (dataset / "labels").as_posix(),
        "records": records, "effect_distribution_by_type": effect_distribution(records), "recovered_transactions": recovered,
        "identity_split_rule": "plate_num is unique across meta.csv; split identities before any future augmentations",
    }
    transaction["status"] = "meta_written"
    transaction["summary"] = summary
    write_transaction(dataset, transaction)
    update_manifest(dataset, summary)
    transaction["status"] = "completed"
    write_transaction(dataset, transaction)
    shutil.rmtree(stage_dir)
    if args.contact_sheet:
        create_contact_sheet(records, dataset, args.contact_sheet.resolve())
        summary["contact_sheet"] = args.contact_sheet.resolve().as_posix()
    if args.examples_dir:
        summary["large_examples"] = create_large_examples(records, dataset, args.examples_dir.resolve(), all_records=args.examples_all)
    if args.summary:
        atomic_write_text(args.summary.resolve(), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("probe", "train", "validation", "test"), required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--examples-dir", type=Path)
    parser.add_argument("--examples-all", action="store_true", help="write a clean crop for every record, for visual diagnostics")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--role", choices=("debug", "train_candidate"), default="debug", help="debug batches are excluded from future training/submission")
    parser.add_argument("--plate-types", nargs="+", choices=tuple(TYPE_SPECS), default=tuple(TYPE_SPECS),
                        help="cyclic type sequence; use type1b alone to replace only that class")
    parser.add_argument("--region-cycle", nargs="+", help="fixed two/three-digit regions for a diagnostic control batch")
    parser.add_argument("--replace-completed", action="store_true", help="replace only this generator's already-completed operation")
    parser.add_argument("--interrupt-after-staged", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--interrupt-after-committed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if not re.fullmatch(r"[a-z0-9_-]+", args.prefix):
        parser.error("--prefix must contain only lowercase ASCII letters, digits, _ or -")
    if args.region_cycle and any(not re.fullmatch(r"\d{2,3}", region) for region in args.region_cycle):
        parser.error("--region-cycle values must contain two or three digits")
    return args


def main() -> int:
    args = parse_args()
    try:
        summary = generate(args)
    except SimulatedInterruption as error:
        print(f"INTERRUPTED_FOR_TEST: {error}")
        return 3
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
