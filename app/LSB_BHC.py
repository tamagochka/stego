import sys
import random

from numpy.typing import NDArray
from numpy import copy, uint8, array, where, sum, abs, pad

from .Embedder import Embedder
from .Extractor import Extractor
from .utils import img_arr_to_vect, img_vect_to_arr, D2B, BE


# значения по умолчанию параметров уникальных для алгоритма
default_msg_len: int = 3


def gen_parity_check_matrix(block_len: int) -> NDArray[uint8]:
    """
    Генерирует матрицу проверки четности.

    Parameters
    ----------
    block_len:int
        размер матрицы
    Returns
    -------
    NDArray[uint8]
        матрица проверки четности
    """

    # размер двоичного представления максимального индекса блока
    block_bits: int = block_len.bit_count()
    tmp: list[NDArray[uint8]] = []
    for i in range(1, block_len + 1):
        tmp.append(D2B(uint8(i), endianness=BE)[8 - block_bits:])
    return array(tmp).T


class LSB_bhc_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения с использованием двоичных кодов Хэмминга (bhc).
    Получает из свойства родителя params параметр работы:
    {'msg_len': 3}
        длинна сообщения в битах, встраиваемого в один блок
    """
    
    def embedding(self):
        if self.cover_object is None: return
        if self.message_bits is None: return

        # получаем параметры работы алгоритма
        msg_len = (self.params or {}).get('msg_len', default_msg_len)

        code = 2

        # получаем покрывающий объект в виде вектор-строки байт
        cover_vect, cover_len, count_lines, count_dim = img_arr_to_vect(self.cover_object)
        if count_lines == 0 or cover_vect is None: return

        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_vect = copy(cover_vect)

        message_len = len(self.message_bits)
        # размер блока в который погружается блок сообщения длинной msg_len
        block_len = code ** msg_len - 1
        
        H = gen_parity_check_matrix(block_len)

        # делаем длинну сообщения кратной msg_len, чтобы распределить его по блокам
        self.message_bits = pad(self.message_bits, (0, msg_len - message_len % msg_len), mode='constant')

        # погружение
        for i, j in zip(range(0, message_len, msg_len), range(0, cover_len, block_len)):
            m_chunk = self.message_bits[i:i + msg_len]
            c_chunk = cover_vect[j:j + block_len] % code

            # сообщение слишком длинное
            if len(c_chunk) != block_len:
                return

            column = (H.dot(c_chunk) - m_chunk) % code
            position = where(sum(abs(H.T - column), axis=1) == 0)[0]

            if len(position) > 0:
                v = int(stego_vect[j:j + block_len][position[0]])
                if v == 255: v = 254
                elif v == 0: v = 1
                else: v += random.choice([1, -1])
                stego_vect[j:j + block_len][position[0]] = v

        self.stego_object = img_vect_to_arr(stego_vect, count_lines, count_dim)



class LSB_bhc_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием двоичных кодов Хэмминга (bhc).
    Получает из свойства родителя params параметр работы:
    {'msg_len': 3}
        длинна сообщения в битах, встраиваемого в один блок
    """
        
    def extracting(self):
        stego_vect, stego_len = img_arr_to_vect(self.stego_object)[0:2]
        if stego_vect is None: return

        # получаем параметры работы алгоритма
        msg_len = (self.params or {}).get('msg_len', default_msg_len)

        code = 2

        # размер блока в который был погружен блок сообщения длинной msg_len
        block_len = code ** msg_len - 1

        H = gen_parity_check_matrix(block_len)

        message_bits: list[uint8] = []
        for i in range(0, stego_len, block_len):
            s_chunk = stego_vect[i:i + block_len] % code
            # если достигли конца стеганограммы
            if len(s_chunk) < block_len:
                break
            m_chunk = H.dot(s_chunk) % code
            message_bits += m_chunk.tolist()

        self.message_bits = array(message_bits)

if __name__ == "__main__":
    sys.exit()
