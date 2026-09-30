"""Первый измеримый вариант: ONNX-детектор + EasyOCR для обрезок."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from plate_text import normalize_plate
from plate_regions import single_line_slope, split_number_region
from square_regions import split_square_fields, split_square_fields_projection


@dataclass
class Detection:
    box: tuple
    confidence: float


def read_image(path):
    # imdecode + fromfile поддерживает русские пути и в Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return frame


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / max(area_a + area_b - overlap, 1e-9)


def suppress_duplicates(detections, threshold):
    selected = []
    for item in sorted(detections, key=lambda x: x.confidence, reverse=True):
        if all(iou(item.box, previous.box) < threshold for previous in selected):
            selected.append(item)
    return selected


def letterbox(frame, size):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor / 255.0, scale, left, top


class OnnxPlateDetector:
    def __init__(self, path, settings):
        import onnxruntime as ort

        if not Path(path).is_file():
            raise FileNotFoundError(f"Нет детектора {path}. Выполни prepare_models.py.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings["cpu_threads"]
        options.inter_op_num_threads = 1
        # CPU-детектор работает одинаково без привязки к CUDA на двух поколениях GPU.
        # Это начальный вариант; окончательная скорость ещё должна быть измерена.
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        if len(shape) != 4 or shape[1] != 3 or inp.type != "tensor(float)":
            raise ValueError(f"Неподдерживаемый вход ONNX: {shape}, {inp.type}")
        self.size = shape[2] if isinstance(shape[2], int) else 640
        if isinstance(shape[3], int) and shape[3] != self.size:
            raise ValueError("Этот адаптер ожидает квадратный вход ONNX.")
        self.name, self.settings = inp.name, settings

    def detect(self, frame):
        tensor, scale, pad_x, pad_y = letterbox(frame, self.size)
        output = self.session.run(None, {self.name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"Неизвестная форма выхода детектора: {output.shape}")
        raw = output[0]
        if raw.shape[0] == 5:
            raw = raw.T
        if raw.shape[1] != 5:
            raise ValueError(f"Ожидался одноклассовый YOLO [N,5], получено {raw.shape}")
        height, width = frame.shape[:2]
        found = []
        for cx, cy, bw, bh, score in raw[raw[:, 4] >= self.settings["detector_confidence"]]:
            x1 = max(0, min(width, (float(cx - bw / 2) - pad_x) / scale))
            y1 = max(0, min(height, (float(cy - bh / 2) - pad_y) / scale))
            x2 = max(0, min(width, (float(cx + bw / 2) - pad_x) / scale))
            y2 = max(0, min(height, (float(cy + bh / 2) - pad_y) / scale))
            if x2 - x1 >= 10 and y2 - y1 >= 8:
                found.append(Detection((x1, y1, x2, y2), float(score)))
        return suppress_duplicates(found, self.settings["nms_iou"])


def infer_plate_type(crop, settings):
    ratio = crop.shape[1] / crop.shape[0]
    if ratio <= settings["square_max_ratio"] and single_line_slope(crop) is None:
        return "type1a"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Слабый тёплый оттенок грязного белого номера ещё не означает жёлтый фон.
    saturation = settings.get("yellow_min_saturation", 110)
    yellow = cv2.inRange(hsv, np.array([12, saturation, 65]), np.array([40, 255, 255]))
    return "type1b" if np.mean(yellow > 0) >= settings["yellow_min_fraction"] else "type1"


def text_band(gray):
    """Вернуть весь fallback-crop без небезопасного выделения text band.

    Вход уже является crop детектора/геометрического поля номера. Любая
    автоматическая граница по components могла отбросить мелкий наклонённый
    регион по x или y. У fallback нет надёжной информации, чтобы отличить
    такой регион от шума, поэтому он консервативно передаёт OCR весь crop и
    не использует текст, разметку либо результат OCR для выбора границ.
    """
    return gray


class PlatePipeline:
    def __init__(self, models, settings, device="auto", ocr_checkpoint=None, format_mode="strict"):
        import easyocr
        import torch

        if format_mode not in ("strict", "extended"):
            raise ValueError(f"Неизвестный режим формата: {format_mode}")
        self.settings, self.format_mode = settings, format_mode
        self.detector = OnnxPlateDetector(Path(models) / "plate_detector.onnx", settings)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Запрошена CUDA, но она недоступна. Запусти check_env.py.")
        self.gpu = device != "cpu" and torch.cuda.is_available()
        torch.set_num_threads(settings["cpu_threads"])
        self.reader = easyocr.Reader(
            ["en"], gpu=self.gpu, detector=False,
            model_storage_directory=str(Path(models) / "easyocr"),
            user_network_directory=str(Path(models) / "easyocr_user"),
            download_enabled=False, verbose=False,
        )
        self.torch = torch
        self.ocr_checkpoint = None
        if ocr_checkpoint is not None:
            checkpoint_path = Path(ocr_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Нет OCR checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path,
                                    map_location="cuda" if self.gpu else "cpu",
                                    weights_only=False)
            if checkpoint.get("format") != "volga_easyocr_finetune_v1":
                raise ValueError("Checkpoint не является внутренним EasyOCR checkpoint проекта")
            if checkpoint.get("character") != self.reader.character:
                raise ValueError("Алфавит OCR checkpoint не совпадает с базовым EasyOCR")
            self.reader.recognizer.load_state_dict(checkpoint["state_dict"], strict=True)
            self.reader.recognizer.eval()
            self.ocr_checkpoint = str(checkpoint_path)

    def synchronize(self):
        if self.gpu:
            self.torch.cuda.synchronize()

    def read_line(self, crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        if crop.size == 0:
            return "", 0.0
        gray = text_band(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        # Один OCR на выделенную строку: второй детектор текста не нужен.
        result = self.reader.recognize(
            gray, detail=1, paragraph=False, decoder="greedy", batch_size=1,
            workers=0, allowlist=allowlist,
            contrast_ths=0.0,
        )
        if not result:
            return "", 0.0
        result = sorted(result, key=lambda r: min(p[0] for p in r[0]))
        return "".join(r[1] for r in result), min(float(r[2]) for r in result)

    def read_fields(self, crop, plate_type="type1"):
        expected_body_length = 5 if plate_type == "type1b" else 6
        bands, details = split_number_region(crop, expected_body_length)
        if bands is None:
            return None, details
        texts, confidences = [], []
        for index, band in enumerate(bands):
            padded = cv2.copyMakeBorder(band, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            result = self.reader.recognize(
                padded, detail=1, paragraph=False, decoder="greedy", batch_size=1,
                workers=0, contrast_ths=0.0,
                allowlist="0123456789" if index else "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texts.append("".join(r[1] for r in result))
            confidences.append(min((float(r[2]) for r in result), default=0.0))
        raw = "".join(texts)
        confidence = min(confidences)
        valid_lengths = (len(texts[0]) == expected_body_length
                         and len(texts[1]) == details["region_components"])
        normalized = normalize_plate(raw, self.settings["allowed_letters"],
                                     self.settings["region_first_digits"], getattr(self, "format_mode", "strict"), plate_type)
        accepted = (valid_lengths and normalized is not None
                    and confidence >= self.settings["ocr_confidence"])
        details.update({"expected_body_length": expected_body_length,
                        "raw_parts": texts, "part_confidences": confidences,
                        "status": "accepted" if accepted else "fallback_to_whole_line"})
        return ((raw, confidence) if accepted else None), details

    def read_square_fields(self, crop):
        bands, details = split_square_fields(crop)
        if bands is None:
            return None, details
        texts, confidences = [], []
        for index, band in enumerate(bands):
            padded = cv2.copyMakeBorder(band, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            result = self.reader.recognize(
                padded, detail=1, paragraph=False, decoder="greedy", batch_size=1,
                workers=0, contrast_ths=0.0,
                allowlist="0123456789" if index == 2 else "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texts.append("".join(r[1] for r in result))
            confidences.append(min((float(r[2]) for r in result), default=0.0))
        raw, confidence = "".join(texts), min(confidences)
        lengths_match = [len(t) for t in texts] == details["component_counts"]
        normalized = normalize_plate(raw, self.settings["allowed_letters"],
                                     self.settings["region_first_digits"], getattr(self, "format_mode", "strict"), "type1a")
        accepted = (lengths_match and normalized is not None
                    and confidence >= self.settings["ocr_confidence"])
        details.update({"raw_parts": texts, "part_confidences": confidences,
                        "status": "accepted" if accepted else "fallback_to_previous_reader"})
        return ((raw, confidence) if accepted else None), details

    def read_square_fields_projection(self, crop):
        """Эксперимент: читать три поля type1a без component-based crop.

        Вызывается только после отклонения обычного пути.  Длины и официальный
        формат проверяются до возврата, поэтому новая ветка не может заменить
        уже принятый результат или пропустить лишний символ.
        """
        parts, details = split_square_fields_projection(crop)
        if parts is None:
            return None, details
        allowlists = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            self.settings["allowed_letters"],
            "0123456789",
        )
        texts, confidences = [], []
        for part, allowlist in zip(parts, allowlists):
            text, confidence = self.read_line(part, allowlist)
            texts.append(text)
            confidences.append(confidence)
        raw, confidence = "".join(texts), min(confidences)
        lengths_match = len(texts[0]) == 4 and len(texts[1]) == 2 and len(texts[2]) in (2, 3)
        normalized = normalize_plate(raw, self.settings["allowed_letters"],
                                     self.settings["region_first_digits"], self.format_mode, "type1a")
        accepted = (lengths_match and normalized is not None
                    and confidence >= self.settings["ocr_confidence"])
        details.update({"raw_parts": texts, "part_confidences": confidences,
                        "expected_lengths": [4, 2, "2_or_3"],
                        "status": "accepted" if accepted else "rejected_fields"})
        return ((raw, confidence) if accepted else None), details

    def predict(self, frame):
        rows, debug = [], []
        for detection in self.detector.detect(frame):
            x1, y1, x2, y2 = detection.box
            crop = frame[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
            plate_type = infer_plate_type(crop, self.settings)
            square_fields = None
            square_details = {"status": "not_square_candidate"}
            if crop.shape[1] / crop.shape[0] <= self.settings["square_max_ratio"]:
                square_fields, square_details = self.read_square_fields(crop)
            if square_fields is not None:
                plate_type = "type1a"
            fields = None
            field_details = {"status": "two_line_layout"}
            if square_fields is None and plate_type != "type1a":
                fields, field_details = self.read_fields(crop, plate_type)
            if square_fields is not None:
                raw, confidence = square_fields
                method = "square_three_fields"
            elif fields is not None:
                raw, confidence = fields
                method = "number_and_region"
            elif plate_type == "type1a":
                # При fallback сохраняем весь detector crop; пограничные
                # символы и регион не должны исчезать вместе с рамкой.
                split = max(1, round(crop.shape[0] * 0.52))
                text1, conf1 = self.read_line(crop[:split])
                text2, conf2 = self.read_line(crop[split:])
                raw, confidence = text1 + text2, min(conf1, conf2)
                method = "two_lines"
            else:
                raw, confidence = self.read_line(crop)
                method = "whole_line"
            text = normalize_plate(raw, self.settings["allowed_letters"],
                                   self.settings["region_first_digits"], self.format_mode, plate_type)
            accepted = text is not None and confidence >= self.settings["ocr_confidence"]
            projection_details = {"status": "disabled"}
            if (not accepted and plate_type == "type1a"
                    and self.settings.get("experimental_two_line_fallback", False)):
                projection, projection_details = self.read_square_fields_projection(crop)
                if projection is not None:
                    raw, confidence = projection
                    text = normalize_plate(raw, self.settings["allowed_letters"],
                                           self.settings["region_first_digits"], self.format_mode, plate_type)
                    accepted = text is not None and confidence >= self.settings["ocr_confidence"]
                    method = "square_projection_three_fields"
            debug.append({"bbox_xyxy": list(detection.box), "raw_ocr": raw,
                          "normalized": text, "plate_type": plate_type,
                          "detector_confidence": detection.confidence,
                          "ocr_confidence": confidence, "accepted": accepted,
                          "ocr_method": method, "format_mode": self.format_mode, "field_attempt": field_details,
                          "square_attempt": square_details,
                          "square_projection_attempt": projection_details})
            if accepted:
                # Рабочая оценка, пока не калиброванная вероятность правильности.
                combined = max(0.0, min(1.0, confidence * detection.confidence))
                rows.append({"plate_num": text, "plate_type": plate_type,
                             "confidence": combined})
        return rows, debug
