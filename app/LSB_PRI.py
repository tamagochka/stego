import os, sys
from random import random

from PIL import Image
from numpy import copy, zeros, uint8, asarray, concatenate, fromfile, array_split, dstack

from .utils import D2B, B2D, chars2bytes, bytes2chars, to_bit_vector, from_bit_vector, step
from .Embedder import Embedder
from .Extractor import Extractor


# значения по умолчанию параметров уникальных для алгоритма
default_start_position: int = 42
default_key: int = 3
default_fill_rest: bool = True


class LSB_PRI_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения с псевдослучайным интервалом между изменяемыми пикселями покрывающего объекта.
    Получает из свойства родителя params параметр работы:
    {'start_position': 42}
        место начала погружения
    {'key': 3}
        ключ, задающий масштабирование шага встраивания (расстояние между битами вложения)
    {'fill_rest': True}
        заполнять незаполненную часть покрывающего объекта случайными битами или нет
    """

    def embeding(self):
        # получаем параметры работы алгоритма
        start_position = default_start_position
        if self.params and 'start_position' in self.params:
            start_position = self.params['start_position']
        key = default_key
        if self.params and 'key' in self.params:
            key = self.params['key']
        fill_rest: bool = default_fill_rest
        if self.params and 'fill_rest' in self.params:
            fill_rest = self.params['fill_rest']
        # получаем цветовые составляющие изображения
        if self.cover_object is None: return
        cover_red = concatenate(self.cover_object[:, :, 0])
        cover_green = concatenate(self.cover_object[:, :, 1])
        cover_blue = concatenate(self.cover_object[:, :, 2])
        # собираем все цветовые составляющие в одну вектор-строку байт
        cover_vect = concatenate([cover_red, cover_green, cover_blue])
        count_lines = len(self.cover_object[:, :, 0])
        cover_len = len(cover_vect)
        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_vect = copy(cover_vect)

        # заполняем покрывающий объект случайными битами, чтобы скрыть место размещения вложоения
        if fill_rest:
            z = 1
            while z <= cover_len:
                b = D2B(cover_vect[z])
                b[0] = round(random())
                stego_vect[z] = B2D(b)
                z += step(D2B(uint8(z & 0xFF)), key)
        
        # погружение
        if self.message_bits is None: return
        message_len = len(self.message_bits)

        z = start_position
        for i in range(message_len):
            b = D2B(cover_vect[z])
            b[0] = self.message_bits[i]
            stego_vect[z] = B2D(b)
            # определяем следующее место встраивания
            z += step(D2B(uint8(z & 0xFF)), key)

        # собираем изображение обратно
        stego_red, stego_green, stego_blue = array_split(stego_vect, 3)
        stego_red = array_split(stego_red, count_lines)
        stego_green = array_split(stego_green, count_lines)
        stego_blue = array_split(stego_blue, count_lines)
        self.stego_object = dstack((stego_red, stego_green, stego_blue))


class LSB_PRI_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с псевдослучайным интервалом между изменяемыми пикселями покрывающего объекта.
    Получает из свойства родителя params параметр работы:
    {'start_position': 42}
        место начала погружения
    {'key': 3}
        ключ, задающий масштабирование шага встраивания (расстояние между битами вложения)
    """

    def extracting(self):
        # получаем параметры работы алгоритма
        start_position = default_start_position
        if self.params and 'start_position' in self.params:
            start_position = self.params['start_position']
        key = default_key
        if self.params and 'key' in self.params:
            key = self.params['key']
        # получаем цветовые составляющие изображения
        if self.stego_object is None: return
        stego_red = concatenate(self.stego_object[:, :, 0])
        stego_green = concatenate(self.stego_object[:, :, 1])
        stego_blue = concatenate(self.stego_object[:, :, 2])
        # собираем все цветовые составляющие в одну вектор-строку байт
        stego_vect = concatenate([stego_red, stego_green, stego_blue])

        stego_len = len(stego_vect)
        # резервируем место под вложение
        self.message_bits = zeros(stego_len, dtype=uint8)
        
        z = start_position
        i = 0
        while z <= stego_len:
            # байт стеганограммы в двоичный вид
            b = D2B(stego_vect[z])
            # сохраняем НЗБ стеганограммы как бит вложения
            self.message_bits[i] = b[0]
            i += 1
            z += step(D2B(uint8(z & 0xFF)), key)


if __name__ == '__main__':
    sys.exit()
