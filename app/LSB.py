import sys
from random import random

from numpy import uint8, copy, zeros

from .utils import D2B, B2D, img_arr_to_vect, img_vect_to_arr
from .Embedder import Embedder
from .Extractor import Extractor


# значения по умолчанию параметров уникальных для алгоритма
default_fill_rest: bool = True


class LSB_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения с использованием непрерывного замещения бит (lsb).
    Получает из свойства родителя params параметр работы:
    {'fill_rest': True}
        заполнять незаполненную часть покрывающего объекта случайными битами или нет
    """

    def embeding(self):
        # получаем параметры работы алгоритма
        fill_rest = (self.params or {}).get('fill_rest', default_fill_rest)

        # получаем покрывающий объект в виде вектор-строки байт
        cover_vect, cover_len, count_lines, count_dim = img_arr_to_vect(self.cover_object)
        if count_lines == 0 or cover_vect is None: return

        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_vect = copy(cover_vect)

        # погружение
        if self.message_bits is None: return
        message_len = len(self.message_bits)
        for i in range(message_len):
            # преобразуем байт покрывающего объекта в двоичный вид
            b = D2B(cover_vect[i])
            # замещаем НЗБ покрывающего объекта битом вложения
            b[0] = self.message_bits[i]
            # сохраняем стеганограмму
            stego_vect[i] = B2D(b)

        # заполняем оставшуюся пустую часть покрывающего объекта случайными битами
        if fill_rest:
            for i in range(message_len, cover_len):
                b = D2B(cover_vect[i])
                b[0] = round(random())
                stego_vect[i] = B2D(b)

        # собираем изображение обратно
        self.stego_object = img_vect_to_arr(stego_vect, count_lines, count_dim)

        # сравнение попиксельно покрывающего объекта и стеганограммы
        # for i in range(len(cover_object)):
        #     for j in range(len(cover_object[i])):
        #         c = ''
        #         for k in range(len(cover_object[i][j])):
        #             c = c + f'{cover_object[i][j][k]:4}'
        #         c = '[' + c + ']'
        #         s = ''
        #         for k in range(len(stego[i][j])):
        #             s = s + f'{stego[i][j][k]:4}'
        #         s = '[' + s + ']'
        #         print(c, ' - ', s, '+' if c != s else ' ')
        #     print()


class LSB_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием непрерывного замещения бит (lsb).
    """

    def extracting(self):
        # получаем стеганограмму в виде вектор-строки байт
        stego_vect, stego_len = img_arr_to_vect(self.stego_object)[0:2]
        if stego_vect is None: return

        # резервируем место под вложение
        self.message_bits = zeros(stego_len, dtype=uint8)
        
        for i in range(stego_len):
            # байт стеганограммы в двоичный вид
            b = D2B(stego_vect[i])
            # сохраняем НЗБ стеганограммы как бит вложения
            self.message_bits[i] = b[0]


if __name__ == '__main__':
    sys.exit()
