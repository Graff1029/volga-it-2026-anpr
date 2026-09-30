"""Короткое или полное внутреннее дообучение существующего EasyOCR CTC.

Новый checkpoint сохраняется отдельно. Исходный models/easyocr/english_g2.pth,
детектор и predict.py этим скриптом не изменяются.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from ocr_finetune_common import (FORMAT_VERSION, batch_tensors, create_reader,
                                 load_checkpoint, load_jsonl, save_json, sha256_file,
                                 state_digest)


class FragmentDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict]):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.root / row["fragment"], row["target_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/training/ocr_synthetic_20260924"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Начать новый эксперимент из совместимого OCR checkpoint без optimizer/step.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=250,
                        help="Атомарно сохранять state+optimizer каждые N шагов; 0 отключает.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    def collate(items):
        paths, texts = zip(*items)
        return batch_tensors(paths), list(texts)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                      collate_fn=collate, generator=generator, drop_last=False)


def make_payload(model, optimizer, reader, base_path: Path, base_sha256: str,
                 args: argparse.Namespace, step: int, losses: list[float],
                 before: str, checkpoint_steps: list[int]) -> dict:
    return {
        "format": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_path.as_posix()),
        "base_sha256": base_sha256,
        "initial_checkpoint": (str(args.init_checkpoint.as_posix()) if args.init_checkpoint else None),
        "easyocr_character": reader.character,
        "character": reader.character,
        "step": step,
        "technical_run_steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "train_config": {"batch_size": args.batch_size, "learning_rate": args.learning_rate,
                         "data": str(args.data.as_posix()),
                         "checkpoint_every": args.checkpoint_every},
        "losses": losses,
        "state_digest_before": before,
        "state_digest_after": state_digest(model.state_dict()),
        "checkpoint_steps": checkpoint_steps,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
    }


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    """Не оставлять частичный .pt: tmp и final находятся на одном томе."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps должен быть положительным")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every не может быть отрицательным")
    if args.output.exists():
        raise FileExistsError(f"Checkpoint уже существует: {args.output}")
    if args.init_checkpoint and args.resume:
        raise ValueError("--init-checkpoint и --resume нельзя использовать вместе")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; технический запуск требует RTX 5070")
    seed_everything(args.seed)
    rows = [row for row in load_jsonl(args.data / "fragments.jsonl") if row["split"] == "train"]
    if not rows:
        raise ValueError("В manifest нет train-фрагментов")
    reader = create_reader(args.models, args.device)
    model = reader.recognizer
    converter = reader.converter
    base_path = args.models / "easyocr" / "english_g2.pth"
    base_sha256 = sha256_file(base_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    start_step = 0
    if args.init_checkpoint:
        load_checkpoint(reader, args.init_checkpoint, args.device)
    elif args.resume:
        checkpoint = load_checkpoint(reader, args.resume, args.device)
        if checkpoint.get("base_sha256") != base_sha256:
            raise ValueError("Checkpoint создан от другого english_g2.pth")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
    model.train()
    before = state_digest(model.state_dict())
    loader = make_loader(FragmentDataset(args.data, rows), args.batch_size, args.seed + start_step)
    iterator = iter(loader)
    losses: list[float] = []
    checkpoint_steps: list[int] = []
    device = torch.device(args.device)
    for offset in range(args.steps):
        try:
            images, texts = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, texts = next(iterator)
        encoded, lengths = converter.encode(texts, batch_max_length=max(map(len, texts)))
        images, encoded, lengths = images.to(device), encoded.to(device), lengths.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=args.device, enabled=args.device == "cuda"):
            prediction = model(images, None)
            # CTCLoss использует float32 даже при CUDA autocast.
            log_probs = functional.log_softmax(prediction.float(), dim=2).permute(1, 0, 2)
            prediction_lengths = torch.full((images.shape[0],), prediction.shape[1],
                                            dtype=torch.long, device=device)
            loss = functional.ctc_loss(log_probs, encoded, prediction_lengths, lengths,
                                       blank=0, reduction="mean", zero_infinity=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Нечисловой loss на шаге {start_step + offset + 1}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        current_step = start_step + offset + 1
        print(f"step={current_step} loss={losses[-1]:.6f}")
        if args.checkpoint_every and current_step % args.checkpoint_every == 0:
            checkpoint_steps.append(current_step)
            save_checkpoint_atomic(args.output, make_payload(
                model, optimizer, reader, base_path, base_sha256, args, current_step,
                losses, before, checkpoint_steps))

    model.eval()
    after = state_digest(model.state_dict())
    payload = make_payload(model, optimizer, reader, base_path, base_sha256, args,
                           start_step + args.steps, losses, before, checkpoint_steps)
    # Последний checkpoint также атомарен, даже если шаг не кратен interval.
    save_checkpoint_atomic(args.output, payload)

    # Отдельный Reader подтверждает strict-загрузку сохранённого checkpoint.
    reloaded = create_reader(args.models, args.device)
    loaded = load_checkpoint(reloaded, args.output, args.device)
    if state_digest(reloaded.recognizer.state_dict()) != after:
        raise AssertionError("Checkpoint загружен, но hash весов не совпал")
    validation_row = next(row for row in load_jsonl(args.data / "fragments.jsonl")
                          if row["split"] == "validation")
    image = __import__("cv2").imread(str(args.data / validation_row["fragment"]), 0)
    result = reloaded.recognize(image, detail=1, paragraph=False, decoder="greedy", batch_size=1,
                                workers=0, contrast_ths=0.0,
                                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    report = {
        "technical_only": True,
        "steps": args.steps,
        "losses": losses,
        "losses_finite": all(np.isfinite(losses)),
        "weights_changed": before != after,
        "state_digest_before": before,
        "state_digest_after": after,
        "checkpoint": str(args.output.as_posix()),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_reload_format": loaded["format"],
        "recognition_smoke": {"fragment": validation_row["fragment"],
                              "target_text": validation_row["target_text"], "raw": result},
        "warning": "Это техническая проверка конечности loss/сохранения/загрузки, не измерение улучшения качества.",
    }
    report_path = args.output.with_suffix(".technical_report.json")
    save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
