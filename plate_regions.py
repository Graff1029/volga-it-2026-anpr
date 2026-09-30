"""Разметка строк и области региона по изображению, без ответов из имён файлов."""

import cv2
import numpy as np


def single_line_slope(crop):
    """Согласованные центры символов помогают отличить наклон от двух строк.

    None означает недостаток признаков, а не доказательство двух строк.
    Правило временное: его ещё надо проверить на настоящих номерах 1а.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    centers = []
    for x, y, w, h, area in stats[1:]:
        if (height * .20 <= h < height * .88 and 2 <= w < h * 1.3
                and area >= 6 and x + w / 2 < width * .72):
            centers.append((x + w / 2, y + h / 2))
    if len(centers) < 4:
        return None
    points = np.float32(centers)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L1, 0, .01, .01).ravel()
    if abs(vx) < .01:
        return None
    slope = float(vy / vx)
    residuals = np.abs(points[:, 1] - (y0 + slope * (points[:, 0] - x0)))
    if abs(slope) > .65 or np.percentile(residuals, 80) > height * .09:
        return None
    return slope


def find_region_separator(gray):
    """Ищем длинную почти вертикальную границу в правой части таблички."""
    height, width = gray.shape
    if height < 12 or width < 40:
        return None
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    choices = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            x, length = (x1 + x2) / 2, abs(y1 - y2)
            if (width * .65 < x < width * .85 and abs(x1 - x2) < height * .23
                    and length > height * .43 and min(y1, y2) < height * .38
                    and max(y1, y2) > height * .64):
                choices.append((float(length), float(x),
                                [float(v) for v in (x1, y1, x2, y2)]))
    return max(choices, key=lambda item: item[0]) if choices else None


def glyph_band(gray, digits_only=False):
    """Выделить крупные символы; мелкие винты, подписи и длинная рамка мешают OCR."""
    if gray.size == 0:
        return None, 0
    height, width = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        max_ratio = 1.1 if digits_only else 1.3
        if (h >= height * .40 and w >= max(2, h * .10)
                and w < h * max_ratio and area >= 6):
            keep.append(index)
    if not keep:
        return None, 0
    clean = gray.copy()
    clean[(labels > 0) & ~np.isin(labels, keep)] = int(np.percentile(gray, 90))
    chosen = stats[keep]
    left, top = min(chosen[:, 0]), min(chosen[:, 1])
    right = max(chosen[:, 0] + chosen[:, 2])
    bottom = max(chosen[:, 1] + chosen[:, 3])
    band = clean[max(0, top - 1):min(height, bottom + 1),
                 max(0, left - 1):min(width, right + 1)]
    return band, len(keep)


def split_number_region(crop, expected_body_count=6):
    """Отдельное чтение основы и региона при известном числе символов основы.

    Если символы слиплись, разметка сомнительна: вызывающий код читает целую
    строку. Подгонки длины текста удалением лишней единицы здесь нет.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    separator = find_region_separator(gray)
    if separator is None:
        return None, {"status": "separator_not_found"}
    _, x, line = separator
    x = round(x)
    gap = max(2, round(width * .012))
    top = max(0, round(height * .03))
    body = gray[top:round(height * .85), max(1, round(width * .02)):x - gap]
    region = gray[top:round(height * .62), x + gap:width - max(1, round(width * .03))]
    body_band, body_count = glyph_band(body)
    region_band, region_count = glyph_band(region, digits_only=True)
    details = {"separator_xyxy": line, "body_components": body_count,
               "region_components": region_count}
    if expected_body_count not in (5, 6):
        raise ValueError("expected_body_count must be 5 (type1b) or 6 (type1/type1a)")
    if body_count != expected_body_count or region_count not in (2, 3):
        return None, {**details, "status": "ambiguous_components"}
    return (body_band, region_band), {**details, "status": "ready"}
