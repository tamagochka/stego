import os
from random import random

from PIL import Image
from numpy import concatenate, asarray, fromfile, uint8, empty, uint16, ndarray, copy, hstack

from utils import chars2bytes, to_bit_vector


def key_pairs_gen(primary_key: int, count_key_pairs: int) -> ndarray[uint8]:
    """
    Генерация вектора пар ключей на основании первичного ключа

    Parameters
    ----------
        primary_key: int
            первичный ключ
        count_key_pairs: int
            количество генерируемых пар ключей
    Returns
    -------
        ndarray[uint8]
            вектор-строка, содержащая key_pairs пар ключей

    """

    key_pairs = empty(2 * count_key_pairs, dtype=uint16)
    key_pairs[0] = primary_key
    for i in range(1, 2 * count_key_pairs):
        key_pairs[i] = int(str(key_pairs[i - 1] ** 2)[:3])
        if key_pairs[i] > 255:
            key_pairs[i] = int(str(key_pairs[i])[0:2])
    return key_pairs.astype(uint8)


def LSB_PRP_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str,
        primary_key: int = 125,
        count_key_pairs: int = 10
):
    """
    Погружение в НЗБ с псевдослучайной перестановкой бит вложения.

    Parameters
    ----------
        cover_file_path: str
            путь к покрывающему объекту
        stego_file_path: str
            путь к стеганограмме
        message_file_path: str
            путь к файлу вложения
        primary_key: int = 125
            первичный ключ
        count_key_pairs: int = 10
            количество пар ключей
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
    message_file_name_bytes = chars2bytes(message_file_name)
    message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
    # собираем все в одину вектор-строку байт
    message_bytes = concatenate((
        message_file_name_bytes_len,
        message_file_name_bytes,
        message_object))
    # преобразуем вктор-строку байт в вектор-строку бит
    message_bits = to_bit_vector(message_bytes)
    message_len = len(message_bits)

    # # получаем цветовые составляющие изображения
    cover_red = cover_object[:, :, 0]
    cover_green = cover_object[:, :, 1]
    cover_blue = cover_object[:, :, 2]

    # # собираем все цветовые составляющие в одну вектор-строку байт
    # cover_vect = concatenate([cover_red, cover_green, cover_blue])
    # count_lines = len(cover_object[:, :, 0])
    # cover_len = len(cover_vect)

    cover_arr = hstack((cover_red, cover_green, cover_blue))
    print(cover_red.shape)
    # стеганограмма - копия покрывающего объекта с измененными НЗБ
    stego_vect = copy(cover_arr)

    X, Y = cover_arr.shape

    print(X, Y)



    # погружение
    # for i in range(message_len):
    #     x = floor(i / Y)