import os, sys
from random import random

from PIL import Image
from numpy import asarray, uint8, fromfile, concatenate, copy, array_split, dstack, zeros
from numpy.typing import NDArray

from .utils import chars2bytes, bytes2chars, to_bit_vector, from_bit_vector, D2B, B2D


# метки начала и конца места погружения вложения в покрывающий объект
default_start_position: int = 42
default_end_label: str = 'k0HEU'
default_key: int = 3


def step(byte: NDArray[uint8], scale: int) -> int:
    """
    Генерация псевдослучайного интервала на основе количества единиц в двоичном представлении номера последнего модифицированного байта изображения умноженного на коэффициент масштабирования.
    
    Parameters
    ----------
    byte: NDArray[uint8]
        двоичная запись числа
    scale: int
        коэффициент масштабирования

    Returns
    -------
    int
        интервал между погруженными битами
    """

    count_bits = 0
    for i in range(len(byte)):
        count_bits += byte[i]
    count_bits = 1 if count_bits == 0 else count_bits
    return int(count_bits * scale)


def LSB_quant_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str,
        start_position: int = default_start_position,
        end_label: str = default_end_label,
        key: int = default_key,
        fill_rest: bool = True
    ):
    """
    Погружение в НЗБ вложения методом квантования.

    Parameters
    ----------
    cover_file_path: str
        имя/путь к покрывающему объекту
    stego_file_path: str
        имя/путь к стеганограмме
    message_file_path: str
        имя/путь к файлу вложения
    start_position: int = 42
        место начала погружения
    end_label: str = 'k0HEU'
        метка конца вложения
    key: int = 3
        шаг погружения
    """

    # загрузка покрывающего объекта
    cover_object = None    
    with Image.open(cover_file_path) as F:
        cover_object = asarray(F, dtype=uint8)

    # загрузка вложения
    message_object = None
    with open(message_file_path, 'rb') as F:
        message_object = fromfile(F, dtype=uint8)
        message_file_name = os.path.basename(F.name)

    # преобразуем имя файла вложения и длинну его имени в байтовые вектор-строки
    end_label_bytes = chars2bytes(end_label)
    message_file_name_bytes = chars2bytes(message_file_name)
    message_file_name_bytes_len = asarray([len(message_file_name_bytes)])

    # собираем все в одину вектор-строку байт
    message_bytes = concatenate((
        message_file_name_bytes_len,
        message_file_name_bytes,
        message_object,
        end_label_bytes))
    # преобразуем вктор-строку байт в вектор-строку бит
    message_bits = to_bit_vector(message_bytes)
    message_len = len(message_bits)

    # получаем цветовые составляющие изображения
    cover_red = concatenate(cover_object[:, :, 0])
    cover_green = concatenate(cover_object[:, :, 1])
    cover_blue = concatenate(cover_object[:, :, 2])
    # собираем все цветовые составляющие в одну вектор-строку байт
    cover_vect = concatenate([cover_red, cover_green, cover_blue])
    count_lines = len(cover_object[:, :, 0])
    cover_len = len(cover_vect)
    # стеганограмма - копия покрывающего объекта с измененными НЗБ
    stego_vect = copy(cover_vect)

    if fill_rest:
        z = uint8(1)
        while z <= cover_len:
            b = D2B(cover_vect[z])
            b[0] = round(random())
            stego_vect[z] = B2D(b)
            z += step(D2B(z), key)

    z = uint8(start_position)
    for i in range(message_len):
        # определяем разницу между двумя смежными пикселями
        # максимальная разница может быть -255 или 255
        diff = abs(int(cover_vect[z]) - int(cover_vect[z - 1]))
        if diff % 2 != message_bits[i]:
            # изменяем разницу между значениями пикселей так, чтобы она соответствовала биту сообщения
            if cover_vect[z] <= cover_vect[z - 1]:
                if int(cover_vect[z]) + 1 <= 255:
                    stego_vect[z] = cover_vect[z] + 1
                else:
                    stego_vect[z] = cover_vect[z] - 1
            else:
                stego_vect[z - 1] = cover_vect[z - 1] + 1
        # определяем следующее место встраивания
        z += step(D2B(z), key)

    # собираем изображение обратно
    stego_red, stego_green, stego_blue = array_split(stego_vect, 3)
    stego_red = array_split(stego_red, count_lines)
    stego_green = array_split(stego_green, count_lines)
    stego_blue = array_split(stego_blue, count_lines)
    stego_object = dstack((stego_red, stego_green, stego_blue))

    with Image.fromarray(stego_object) as F:
        F.save(stego_file_path)


def LSB_quant_extracting(
        stego_file_path: str,
        extract_file_path: str,
        start_position: int = default_start_position,
        end_label: str = default_end_label,
        key: int = default_key
    ):
    """
    Извлечение из НЗБ вложения, погруженного методом квантования.

    Parameters
    ----------
    stego_file_path: str
        имя/путь к стеганограмме
    extract_file_path: str
        путь к файлу вложения (только директория)
    start_position: int = 42
        место начала погружения
    end_label: str = 'k0HEU'
        метка конца места вложения
    key: int = 3
        шаг погружения
    """

    # загрузка стеганограммы
    stego_object = None
    with Image.open(stego_file_path, 'r') as F:
        stego_object = asarray(F, dtype=uint8)

    # получаем цветовые составляющие изображения
    stego_red = concatenate(stego_object[:, :, 0])
    stego_green = concatenate(stego_object[:, :, 1])
    stego_blue = concatenate(stego_object[:, :, 2])
    # собираем все цветовые составляющие в одну вектор-строку байт
    stego_vect = concatenate([stego_red, stego_green, stego_blue])

    stego_len = len(stego_vect)
    # резервируем место под вложение
    message_bits = zeros(stego_len, dtype=uint8)

    z = uint8(start_position)
    i = 0
    while z <= stego_len:
        diff = abs(int(stego_vect[z]) - int(stego_vect[z - 1]))
        message_bits[i] = diff % 2
        z += step(D2B(z), key)
        i += 1

    # выделяем вложение
    message_bytes = from_bit_vector(message_bits)
    message = bytes2chars(message_bytes)
    message = message[0:message.find(end_label)]

    message_file_name_len = chars2bytes(message[0])[0]
    message_file_name = message[1:message_file_name_len + 1]

    message_file_path = os.path.join(extract_file_path, message_file_name)
    with open(message_file_path, 'bw') as F:
        F.write(chars2bytes(message[message_file_name_len + 1:]))



if __name__ == '__main__':
    sys.exit()
