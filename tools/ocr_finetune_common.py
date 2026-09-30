"""Общие функции для внутреннего дообучения EasyOCR.

Этот модуль не подключается из predict.py: базовый ``english_g2.pth`` остаётся
рабочим весом распознавателя.  Чекпойнты дообучения загружаются только явно
утилитами этого каталога.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image


FORMAT_VERSION = "volga_easyocr_finetune_v1"
MODEL_HEIGHT = 64  # EasyOCR 1.7.2: easyocr.easyocr.imgH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def create_reader(models_dir: Path, device: str):
    """Создать Reader тем же способом, что и PlatePipeline, без скачиваний."""
    import easyocr

    use_gpu = device == "cuda"
    return easyocr.Reader(
        ["en"], gpu=use_gpu, detector=False,
        model_storage_directory=str(models_dir / "easyocr"),
        user_network_directory=str(models_dir / "easyocr_user"),
        download_enabled=False, verbose=False,
    )


def state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Хеш состояния для проверки, что хотя бы один вес действительно изменился."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_checkpoint(reader, checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format") != FORMAT_VERSION:
        raise ValueError(f"Неподдерживаемый checkpoint: {checkpoint.get('format')!r}")
    if checkpoint.get("character") != reader.character:
        raise ValueError("Алфавит checkpoint не совпадает с активным EasyOCR")
    reader.recognizer.load_state_dict(checkpoint["state_dict"], strict=True)
    reader.recognizer.eval()
    return checkpoint


def easyocr_tensor(image: np.ndarray) -> torch.Tensor:
    """В точности повторить resize -> AlignCollate -> NormalizePAD EasyOCR.

    ``get_image_list`` сначала меняет размер вертикальных фрагментов так, что
    их промежуточная высота может быть больше 64. Затем ``AlignCollate``
    *всегда* приводит изображение к ``imgH=64``. Ранняя версия пропустила
    второй шаг для вертикальных фрагментов, что и давало высоту 67 в batch.
    Здесь оба шага сохранены намеренно; возвращаемая тензорная высота всегда
    ``MODEL_HEIGHT``.
    """
    if image is None or image.size == 0:
        raise ValueError("Пустой OCR-фрагмент")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Нулевой размер OCR-фрагмента")
    ratio = width / height
    # 1) easyocr.utils.compute_ratio_and_resize, используемый get_image_list.
    if ratio < 1.0:
        inverse_ratio = 1.0 / ratio
        intermediate = cv2.resize(image,
                                  (MODEL_HEIGHT, max(1, int(MODEL_HEIGHT * inverse_ratio))),
                                  interpolation=cv2.INTER_LINEAR)
    else:
        intermediate = cv2.resize(image,
                                  (max(1, int(MODEL_HEIGHT * ratio)), MODEL_HEIGHT),
                                  interpolation=cv2.INTER_LINEAR)
    # 2) easyocr.recognition.AlignCollate(imgH=64, keep_ratio_with_pad=True).
    inter_height, inter_width = intermediate.shape[:2]
    final_width = max(1, math.ceil(MODEL_HEIGHT * inter_width / inter_height))
    resized = np.asarray(Image.fromarray(intermediate).resize(
        (final_width, MODEL_HEIGHT), Image.Resampling.BICUBIC), dtype=np.uint8)
    if resized.shape != (MODEL_HEIGHT, final_width):
        raise AssertionError(f"EasyOCR resize дал неожиданную форму {resized.shape}")
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    return tensor.sub(0.5).div(0.5).unsqueeze(0)


def batch_tensors(paths: Iterable[Path]) -> torch.Tensor:
    tensors = [easyocr_tensor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)) for path in paths]
    if not tensors:
        raise ValueError("Нельзя собрать пустой OCR-batch")
    if any(tensor.shape[1] != MODEL_HEIGHT for tensor in tensors):
        raise AssertionError("OCR-фрагмент не приведён к MODEL_HEIGHT до padding")
    max_width = max(tensor.shape[2] for tensor in tensors)
    result = []
    for tensor in tensors:
        if tensor.shape[2] < max_width:
            padding = tensor[:, :, -1:].expand(1, MODEL_HEIGHT, max_width - tensor.shape[2])
            tensor = torch.cat((tensor, padding), dim=2)
        result.append(tensor)
    batch = torch.stack(result, dim=0)
    if batch.shape[2] != MODEL_HEIGHT:
        raise AssertionError(f"Неверная высота OCR-batch: {tuple(batch.shape)}")
    return batch


def cer(reference: str, hypothesis: str) -> int:
    """Расстояние Левенштейна; делитель вычисляет вызывающий код."""
    row = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, 1):
        next_row = [i]
        for j, target in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (source != target)))
        row = next_row
    return row[-1]
