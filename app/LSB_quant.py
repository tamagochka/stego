import sys
from random import random

from numpy import uint8, copy, zeros

from .utils import D2B, B2D, step, img_arr_to_vect, img_vect_to_arr
from .Embedder import Embedder
from .Extractor import Extractor


# значения по умолчанию параметров уникальных для алгоритма
default_start_position: int = 42
default_key: int = 3
default_fill_rest: bool = True


class LSB_quant_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения методом квантования (quant).
    Получает из свойства родителя params параметр работы:
    {'start_position': 42}
        место начала погружения
    {'key': 3}
        ключ, задающий масштабирование шага погружения (расстояние между битами вложения)
    {'fill_rest': False}
        заполнять незаполненную часть покрывающего объекта случайными битами или нет
    """

    def embedding(self):
        # получаем параметры работы алгоритма
        start_position = (self.params or {}).get('start_position', default_start_position)
        key = (self.params or {}).get('key', default_key)
        fill_rest = (self.params or {}).get('fill_rest', default_fill_rest)

        # получаем цветовые составляющие изображения
        cover_vect, cover_len, count_lines, count_dim = img_arr_to_vect(self.cover_object)
        if count_lines == 0 or cover_vect is None: return

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
            # определяем разницу между двумя смежными пикселями
            # максимальная разница может быть -255 или 255
            diff = abs(int(cover_vect[z]) - int(cover_vect[z - 1]))
            if diff % 2 != self.message_bits[i]:
                # изменяем разницу между значениями пикселей так, чтобы она соответствовала биту сообщения
                if cover_vect[z] <= cover_vect[z - 1]:
                    if int(cover_vect[z]) + 1 <= 255:
                        stego_vect[z] = cover_vect[z] + 1
                    else:
                        stego_vect[z] = cover_vect[z] - 1
                else:
                    stego_vect[z - 1] = cover_vect[z - 1] + 1
            # определяем следующее место погружения
            z += step(D2B(uint8(z & 0xFF)), key)

        # собираем изображение обратно
        self.stego_object = img_vect_to_arr(stego_vect, count_lines, count_dim)


class LSB_quant_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного методом квантования (quant).
    Получает из свойства родителя params параметр работы:
    {'start_position': 42}
        место начала погружения
    {'key': 3}
        ключ, задающий масштабирование шага погружения (расстояние между битами вложения)
    """

    def extracting(self):
        # получаем параметры работы алгоритма
        start_position = (self.params or {}).get('start_position', default_start_position)
        key = (self.params or {}).get('key', default_key)

        # получаем стеганограмму в виде вектор-строки байт
        stego_vect, stego_len = img_arr_to_vect(self.stego_object)[0:2]
        if stego_vect is None: return
        
        # резервируем место под вложение
        self.message_bits = zeros(stego_len, dtype=uint8)
        z = start_position
        i = 0
        while z <= stego_len:
            diff = abs(int(stego_vect[z]) - int(stego_vect[z - 1]))
            self.message_bits[i] = diff % 2
            z += step(D2B(uint8(z & 0xFF)), key)
            i += 1


if __name__ == '__main__':
    sys.exit()
