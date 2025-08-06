import sys
from math import floor, ceil
from random import randrange

from numpy import uint8, hstack, copy, dstack, zeros

from .utils import D2B
from .Embedder import Embedder
from .Extractor import Extractor


class LSB_block_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения с использованием блочного погружения для большей стойкости к изменениям покрывающего объекта.
    """

    def embeding(self):
        # получаем цветовые составляющие изображения
        if self.cover_object is None: return
        cover_red = self.cover_object[:, :, 0]
        cover_green = self.cover_object[:, :, 1]
        cover_blue = self.cover_object[:, :, 2]
        # соединяем двумерные цветовые плоскости в один двумерный массив
        cover_arr = hstack((cover_red, cover_green, cover_blue))
        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_arr = copy(cover_arr)
        # размеры массива
        X, Y = stego_arr.shape

        # погружение
        if self.message_bits is None: return
        message_len = len(self.message_bits)
        # округляем в большую сторону количество бит погружаемых в строку
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
                if even != self.message_bits[y + i * Y]:
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
        self.stego_object = dstack((stego_red, stego_green, stego_blue))


class LSB_block_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием блочного погружения.
    """
    
    def extracting(self):
        # получаем цветовые составляющие изображения
        if self.stego_object is None: return
        stego_red = self.stego_object[:, :, 0]
        stego_green = self.stego_object[:, :, 1]
        stego_blue = self.stego_object[:, :, 2]
        # соединяем двумерные цветовые плоскости в один двумерный массив
        stego_arr = hstack((stego_red, stego_green, stego_blue))

        # размеры массива
        X, Y = stego_arr.shape
        # определяем количество бит в строке
        bit_per_line = stego_arr[stego_arr.shape[0] - 1, stego_arr.shape[1] - 1]
        # длинна вложения
        message_len = Y * int(bit_per_line)
        # резервируем место под битовую вектор-строку вложения
        self.message_bits = zeros(message_len, dtype=uint8)

        for i in range(bit_per_line):
            # начало и конец блока
            block_start = i * floor(X / bit_per_line)
            block_end = (i + 1) * floor(X / bit_per_line)
            for y in range(Y):
                block = stego_arr[block_start:block_end, y]
                even = 0
                for x in range(len(block)):
                    even = even ^ D2B(block[x])[0]
                self.message_bits[y + i * Y] = even


if __name__ == '__main__':
    sys.exit()
    