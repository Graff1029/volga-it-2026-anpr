"""Правила номера из присланного задания. Никаких ответов по именам файлов."""

import re

LETTERS = "ABEKMHOPCTYX"
CYRILLIC = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2"}
TO_LETTER = {"0": "O", "8": "B"}


def clean_text(text):
    # Убираем разделители, но не удаляем произвольные символы внутри номера.
    text = text.upper().translate(CYRILLIC)
    text = re.sub(r"[\s.\-]", "", text)
    if text.endswith("RUS"):
        text = text[:-3]
    return text


def _position_classes(plate_type):
    """Официальные позиции букв и цифр без кода региона.

    Неизвестный тип оставляем совместимым с историческим однострочным
    форматом: в inference тип уже известен до этой проверки.
    """
    if plate_type == "type1b":
        return (0, 1), 5
    return (0, 4, 5), 6


def normalize_plate(text, letters=LETTERS, region_first_digits="", format_mode="strict", plate_type=None):
    """Вернуть номер либо None, не дополняя и не удаляя символы.

    Оба поддерживаемых имени режима следуют официальным маскам: type1/type1a
    ``LDDDLLRR(R)`` и type1b ``LLDDDRR(R)``. ``extended`` сохранён только как
    обратносуместимый параметр CLI, а не как иной контракт формата.
    """
    text = clean_text(text)
    if format_mode not in ("strict", "extended"):
        raise ValueError(f"Неизвестный режим формата: {format_mode}")
    letter_positions, body_length = _position_classes(plate_type)
    if len(text) not in (body_length + 2, body_length + 3):
        return None
    result = []
    for pos, char in enumerate(text):
        is_letter = pos in letter_positions
        if char == "#":
            result.append(char)
            continue
        if is_letter:
            char = TO_LETTER.get(char, char)
            if char not in letters:
                return None
        else:
            char = TO_DIGIT.get(char, char)
            if char not in "0123456789":
                return None
        result.append(char)
    value = "".join(result)
    # ``region_first_digits`` remains an ignored compatibility argument for
    # callers with old settings.  It cannot reintroduce a non-official gate:
    # official regions have any two or three digits (e.g. 323, 550 and 778).
    return value


def validate_plate_literal(text, letters=LETTERS, format_mode="strict", plate_type=None):
    """Проверка готовой разметки без молчаливого исправления ошибок."""
    if format_mode not in ("strict", "extended"):
        raise ValueError(f"Неизвестный режим формата: {format_mode}")
    letter_positions, body_length = _position_classes(plate_type)
    if len(text) not in (body_length + 2, body_length + 3):
        return False
    return all(c == "#" or c in (letters if i in letter_positions else "0123456789")
               for i, c in enumerate(text))
