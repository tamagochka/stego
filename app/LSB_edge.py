import sys

from numpy import dstack, copy, array, uint8

from .Embedder import Embedder
from .Extractor import Extractor
from .utils import edge_detect, img_arr_surfs_to_one_arr, one_arr_to_img_arr_surfs, B2D, D2B


# значения по умолчанию параметров уникальных для алгоритма
default_max_capacity: int = 3


class LSB_edge_embedding(Embedder):
    """
    Реализация алгоритма контурного погружения в НЗБ вложения с использованием непрерывного замещения бит (edge).
    Получает из свойства родителя params параметр работы:
    {'max_capacity': 3}
        количество НЗБ пикселя, которые могут быть изменены в зависимости от значения оператора Прюитт для данного пикселя
    """

    def embedding(self):
        if self.cover_object is None: return

        # получаем параметры работы алгоритма
        max_capacity = (self.params or {}).get('max_capacity', default_max_capacity)

        # отсекаем max_capacity НЗБ, для того чтобы они не влияли на вычисление оператора Прюитт
        tmp = (self.cover_object // (2 ** max_capacity)) * (2 ** max_capacity)
        # применяем оператор Прюитт
        edged_surfs = edge_detect(tmp)

        if edged_surfs is None: return
        # собираем цветовые плоскости (если они есть), содержащие контур изображения, в один массив
        if len(edged_surfs) > 1:
            edged_surfs = dstack(edged_surfs)
        else:
            edged_surfs = edged_surfs[0]
        # соединяем двумерные цветовые плоскости в один двумерный массив
        cover_arr, *widths = img_arr_surfs_to_one_arr(self.cover_object)
        if cover_arr is None: return
        # соединяем двумерные цветовые плоскости, содержащие контур изображения, в один двумерный массив
        edge_arr, *_ = img_arr_surfs_to_one_arr(edged_surfs)
        if edge_arr is None: return
        
        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_arr = copy(cover_arr)

        if self.message_bits is None: return
        message_len = len(self.message_bits)
        bit_index = 0

        # погружение
        for x in range(edge_arr.shape[0]):
            for y in range(edge_arr.shape[1]):
                if bit_index >= message_len:
                    break
                # в зависимости от того насколько пиксель находится на контуре определяем сколько бит может быть погружено в пиксель
                bits_cap = int(edge_arr[x, y] // ((255 // (max_capacity + 1)) + 1))
                # ничего не погружаем
                if bits_cap == 0:
                    continue
                bits_left = message_len - bit_index
                bits2write = min(bits_cap, bits_left)
                bits = self.message_bits[bit_index:bit_index + bits2write]
                bit_index += bits2write
                stego_arr[x, y] = (stego_arr[x, y] & (0xFF ^ ((1 << bits2write) - 1))) | B2D(bits)

        self.stego_object = one_arr_to_img_arr_surfs(stego_arr, *widths)


class LSB_edge_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием алгоритма контурного погружения в НЗБ (edge).
    Получает из свойства родителя params параметр работы:
    {'max_capacity': 3}
        количество НЗБ пикселя, которые могут быть изменены в зависимости от значения оператора Прюитт для данного пикселя
    """

    def extracting(self):
        if self.stego_object is None: return

        # получаем параметры работы алгоритма
        max_capacity = (self.params or {}).get('max_capacity', default_max_capacity)

        # отсекаем max_capacity НЗБ, для того чтобы они не влияли на вычисление оператора Прюитт
        tmp = (self.stego_object // (2 ** max_capacity)) * (2 ** max_capacity)
        # применяем оператор Прюитт
        edged_surfs = edge_detect(tmp)

        if edged_surfs is None: return
        # собираем цветовые плоскости (если они есть), содержащие контур изображения, в один массив
        if len(edged_surfs) > 1:
            edged_surfs = dstack(edged_surfs)
        else:
            edged_surfs = edged_surfs[0]
        # соединяем двумерные цветовые плоскости в один двумерный массив
        stego_arr, *_ = img_arr_surfs_to_one_arr(self.stego_object)
        if stego_arr is None: return
        # соединяем двумерные цветовые плоскости, содержащие контур изображения, в один двумерный массив
        edge_arr, *_ = img_arr_surfs_to_one_arr(edged_surfs)
        if edge_arr is None: return

        # битовая вектор-строка вложения
        message_bits: list[int] = []

        for x in range(edge_arr.shape[0]):
            for y in range(edge_arr.shape[1]):
                # в зависимости от того насколько пиксель находится на контуре определяем сколько бит было погружено в пиксель
                bits_cap = int(edge_arr[x, y] // ((255 // (max_capacity + 1)) + 1))
                if bits_cap == 0:
                    continue
                bits = D2B(uint8(stego_arr[x, y] & (1 << bits_cap) - 1))
                for i in range(bits_cap):
                    message_bits.append(bits[i])

        self.message_bits = array(message_bits, dtype=uint8)


if __name__ == "__main__":
    sys.exit()
    