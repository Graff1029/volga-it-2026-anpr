"""Области номера 1а: четыре символа сверху, две буквы и регион снизу.

Это геометрические эвристики для исходного решения, не обученная модель.
Ответы, имена файлов и проверочная разметка здесь не используются.
"""

import cv2
import numpy as np


def _hide_side_frame(gray):
    height, width = gray.shape
    result = gray.copy()
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return result
    for side in ("left", "right"):
        choices = []
        for x1, y1, x2, y2 in lines[:, 0]:
            x, length = (x1 + x2) / 2, abs(y1 - y2)
            at_edge = x < width * .12 if side == "left" else x > width * .88
            if at_edge and abs(x1 - x2) < height * .18 and length > height * .40:
                choices.append((length, x1, y1, x2, y2))
        if not choices:
            continue
        _, x1, y1, x2, y2 = max(choices, key=lambda item: item[0])
        gap = max(1, round(width * .012))
        for y in range(height):
            x = round(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
            # Ограничение не даёт линии рамки стереть центральную часть номера.
            if side == "left":
                result[y, :min(round(width * .15), max(0, x + gap))] = 255
            else:
                result[y, max(round(width * .85), min(width, x - gap)):] = 255
    return result


def _field_glyphs(gray, expected_counts, region=False):
    height, width = gray.shape
    if min(height, width) < 3:
        return None, 0
    background = np.percentile(gray, 85, axis=1).astype(np.float32)
    normalized = np.clip(gray.astype(np.float32) * 235
                         / np.maximum(background[:, None], 30), 0, 255).astype(np.uint8)
    _, global_mask = cv2.threshold(normalized, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    block = max(3, min(51, (min(height, width) // 2) * 2 - 1))
    adaptive = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block, 7)
    candidates = []
    for source, mask in enumerate((global_mask, adaptive)):
        mask = mask.copy()
        mask[0, :] = mask[-1, :] = 0
        mask[:, 0] = mask[:, -1] = 0
        horizontal = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((1, max(3, round(width * .75))), np.uint8))
        mask[horizontal > 0] = 0
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = []
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            min_width = .18 if region else .10
            if h < height * .35 or w < max(2, h * min_width) or w > h * 1.3 or area < 5:
                continue
            if h > height * .70 and w < h * .22 and (x < width * .1 or x + w > width * .9):
                continue
            if w > width * .9 and h > height * .85:
                continue
            keep.append(index)
        if region and len(keep) >= 3:
            median_height = float(np.median(stats[keep, 3]))
            # Высокая правая рамка, соединённая с флагом, не является цифрой.
            keep = [i for i in keep if not (
                stats[i, 3] > median_height * 1.3
                and stats[i, 0] + stats[i, 2] / 2 > width * .65)]
        if not keep:
            continue
        chosen = stats[keep]
        x1, y1 = min(chosen[:, 0]), min(chosen[:, 1])
        x2 = max(chosen[:, 0] + chosen[:, 2])
        y2 = max(chosen[:, 1] + chosen[:, 3])
        clean = np.where(np.isin(labels, keep), normalized, 255).astype(np.uint8)
        band = clean[max(0, y1 - 1):min(height, y2 + 1),
                     max(0, x1 - 1):min(width, x2 + 1)]
        score = (len(keep) in expected_counts,
                 -min(abs(len(keep) - n) for n in expected_counts), -source)
        candidates.append((score, band, len(keep)))
    if not candidates:
        return None, 0
    _, band, count = max(candidates, key=lambda item: item[0])
    return band, count


def split_square_fields(crop):
    """Отказаться от нового способа, если структура 4 + 2 + (2/3) не видна."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if height < 16 or width < 24:
        return None, {"status": "crop_too_small"}
    gray = _hide_side_frame(gray)
    split, separator = round(height * .5), round(width * .53)
    lower = gray[split:]
    lines = (cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(lower)[0]
             if min(lower.shape) >= 12 else None)
    choices = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            x, length = (x1 + x2) / 2, abs(y1 - y2)
            if (width * .44 < x < width * .68 and length > height * .23
                    and abs(x1 - x2) < height * .14):
                choices.append((float(length), float(x)))
    if choices:
        separator = round(max(choices)[1])
    pad, gap = max(1, round(width * .06)), max(1, round(width * .012))
    parts = [gray[max(1, round(height * .04)):split, pad:width - pad],
             gray[split:round(height * .92), pad:separator - gap],
             gray[split:round(height * .84), separator + gap:width - pad]]
    bands, counts = [], []
    for index, part in enumerate(parts):
        expected = (4,) if index == 0 else (2,) if index == 1 else (2, 3)
        band, count = _field_glyphs(part, expected, region=index == 2)
        bands.append(band)
        counts.append(count)
    details = {"component_counts": counts, "separator_x": separator}
    if counts[0] != 4 or counts[1] != 2 or counts[2] not in (2, 3):
        return None, {**details, "status": "ambiguous_components"}
    return bands, {**details, "status": "ready"}


def split_square_fields_projection(crop):
    """Разделить type1a по световым проекциям, не выделяя компоненты.

    Это запасной эксперимент только для уже отклонённого двухстрочного
    результата.  ``crop`` не очищается: границы строк и полей выбираются по
    маске для измерения, а в OCR передаются исходные пиксели.  Если долина
    между строками или вертикальный разделитель не выражены, функция честно
    отказывается вместо угадывания координат.
    """
    if crop.ndim != 3:
        raise ValueError("Ожидался цветной исходный crop type1a")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if height < 20 or width < 36:
        return None, {"status": "crop_too_small"}
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_rows = np.mean(mask > 0, axis=1)
    ink_cols = np.mean(mask > 0, axis=0)

    def valley(values, start, end, margin):
        """Вернуть глубокую локальную долину либо None без подбора OCR."""
        if end - start < 3:
            return None
        index = start + int(np.argmin(values[start:end]))
        left = values[max(0, start - margin):max(start, index - margin)]
        right = values[min(end, index + margin):min(len(values), end + margin)]
        if not len(left) or not len(right):
            return None
        side_activity = min(float(np.median(left)), float(np.median(right)))
        # Линия считается границей только при заметном провале относительно
        # обеих соседних областей; это не критерий числа символов.
        if side_activity < .025 or values[index] > side_activity * .70:
            return None
        return index

    row_margin = max(2, round(height * .08))
    row_split = valley(ink_rows, round(height * .30), round(height * .70), row_margin)
    if row_split is None:
        return None, {"status": "ambiguous_row_gap"}
    col_margin = max(2, round(width * .06))
    separator = valley(ink_cols, round(width * .40), round(width * .76), col_margin)
    if separator is None:
        return None, {"status": "ambiguous_field_separator", "row_split": row_split}
    if separator < width * .35 or separator > width * .80:
        return None, {"status": "ambiguous_field_separator", "row_split": row_split}

    # Каждому OCR передаём исходный crop своего поля: ни frame, ни символы
    # connected-components не удаляются и не подменяются.
    parts = (crop[:row_split], crop[row_split:, :separator], crop[row_split:, separator:])
    if any(part.size == 0 for part in parts):
        return None, {"status": "empty_projection_field", "row_split": row_split,
                      "separator_x": separator}
    return parts, {"status": "ready", "row_split": row_split,
                    "separator_x": separator, "method": "ink_projection_no_components"}
