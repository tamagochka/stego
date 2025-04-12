import os
from math import floor, ceil
from random import randrange

from PIL import Image
from numpy import asarray, uint8, fromfile, concatenate, hstack, copy, dstack

from utils import chars2bytes, bytes2chars, to_bit_vector, from_bit_vector, D2B, B2D


def LSB_block_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str
    ):
    """
    Блочное погружение в НЗБ для большей стойкости к изменениям покрывающего объекта

    Parameters
    ----------
        cover_file_path: str
            путь к покрывающему объекту
        stego_file_path: str
            путь к стеганограмме
        message_file_path: str
            путь к файлу вложения
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

    # получаем цветовые составляющие изображения
    cover_red = cover_object[:, :, 0]
    cover_green = cover_object[:, :, 1]
    cover_blue = cover_object[:, :, 2]

    # соединяем двумерные цветовые плоскости в один двумерный массив
    cover_arr = hstack((cover_red, cover_green, cover_blue))

    # стеганограмма - копия покрывающего объекта с измененными НЗБ
    stego_arr = copy(cover_arr)
    # размеры массива
    X, Y = stego_arr.shape

    # округляем в большую сторону
    # количество бит погружаемых в строку
    bit_per_line = ceil(message_len / Y)

    for i in range(bit_per_line):
        # начало и конец блока
        block_start = i * floor(X / bit_per_line)
        block_end = (i + 1) * floor(X / bit_per_line)
        for y in range(Y):
            # если вложение закончилось, то выходим
            if y + i * Y >= message_len:
                break
            # вырезаем блок
            block = stego_arr[block_start:block_end, y]
            # определяем четность блока
            even = 0
            for x in range(len(block)):
                even = even ^ D2B(block[x])[0]
            # если четность блока не совпадает с четностью бит вложения
            # то меняем произвольный бит в блоке
            if even != message_bits[y + i * Y]:
                n = ceil(randrange(len(block)))
                if block[n] % 2 == 0:
                    block[n] += 1
                else:
                    block[n] -= 1
                # помещаем измененный блок обратно
                stego_arr[block_start:block_end, y] = block

    stego_arr[stego_arr.shape[0] - 1, stego_arr.shape[1] - 1] = bit_per_line

    # собираем обратно изображение из цветовых составляющих
    stego_red = stego_arr[:, :cover_blue.shape[1]]
    stego_green = stego_arr[:, cover_blue.shape[1]: cover_blue.shape[1] + cover_green.shape[1]]
    stego_blue = stego_arr[:, cover_blue.shape[1] + cover_green.shape[1]: cover_blue.shape[1] + cover_green.shape[1] + cover_red.shape[1]]
    stego_object = dstack((stego_red, stego_green, stego_blue))

    with Image.fromarray(stego_object) as F:
        F.save(stego_file_path)


def LSB_block_extracting(
        stego_file_path: str,
        extract_file_path: str
    ):
    """
    
    """
    pass