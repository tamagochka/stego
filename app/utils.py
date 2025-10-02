import sys
from typing import Any

from numpy.typing import NDArray
from numpy import zeros, array, uint8, uint16, concatenate, array_split, dstack, hstack


class MersenneTwister(object):
    """
    Генератор псевдослучайных чисел на основе алгоритма "Вихрь Мерсена".

    Attributes
    ----------
    seed_val: int = 0xDEADBEEF
        значение зерна, используемого при генерации псевдослучайных чисел
    
    Methods
    -------
    set_seed(seed: int=0xDEADBEEF)
        Задать зерно для генерации случайных чисел.
    get_seed() -> int
        Получить текущее значение зерна.
    extract_number() -> int
        Генерирует псевдо-случайное целое число в диапазоне [0..4294967296).
    random() -> float
        Генерирует псевдо-случайное рациональное число в диапазоне [0..1).
    randint(range_values: int | tuple[int, int]=(1 << 32)) -> int
        Генерирует массив NDArray псевдо-случайных целых чисел заданной формы в заданном диапазоне [a..b).
    randarrint(shape: int | tuple[int, ...], range_values: int | tuple[int, int]=(1 << 32)) -> NDArray
        Генерирует массив NDArray псевдо-случайных целых чисел заданной формы в заданном диапазоне [a..b).
    """

    n: int = 624
    m: int = 397
    a: int = 0x9908B0DF
    b: int = 0x9D2C5680
    c: int = 0xEFC60000
    u: int = 11
    s: int = 7
    t: int = 15
    l: int = 18

    seed_val: int = 0xDEADBEEF


    def __init__(self, seed: int=0xDEADBEEF):
        self.MT = [0] * self.n
        self.index = 0
        self.set_seed(seed)


    def set_seed(self, seed: int=0xDEADBEEF):
        """
        Задать зерно для генерации случайных чисел.

        Parameters
        ----------
        seed: int
            зерно
        """

        self.seed_val = seed
        self.MT[0] = seed & 0xFFFFFFFF
        for i in range(1, self.n):
            self.MT[i] = (1812433253 * (self.MT[i - 1] ^ self.MT[i - 1] >> 30)) + i
            self.MT[i] &= 0xFFFFFFFF
        self.index = self.n


    def get_seed(self) -> int:
        """
        Получить текущее значение зерна.

        Returns
        -------
        int
            текущее значение зерна
        """
        return self.seed_val


    def extract_number(self) -> int:
        """
        Генерирует псевдо-случайное целое число в диапазоне [0..4294967296).

        Returns
        -------
        int
            псевдо-случайное целое число
        """

        if self.index >= self.n:
            self._twist()
        y = self.MT[self.index]
        y ^= (y >> self.u)
        y ^= ((y << self.s) & self.b)
        y ^= ((y << self.t) & self.c)
        y ^= (y >> self.l)
        self.index += 1
        return y & 0xFFFFFFFF
    

    def _twist(self):
        for i in range(self.n):
            y = (self.MT[i] & 0x80000000) + (self.MT[(i + 1) % self.n] & 0x7FFFFFFF)
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ (y >> 1)
            if y % 2 != 0:
                self.MT[i] ^= self.a
        self.index = 0


    def random(self) -> float:
        """
        Генерирует псевдо-случайное рациональное число в диапазоне [0..1).

        Returns
        -------
        float
            псевдо-случайное рациональное число
        """

        return self.extract_number() / 4294967296.0  # нормирование к диапазону [0.. 1)


    def randint(self, range_values: int | tuple[int, int]=(1 << 32)) -> int:
        """
        Генерирует псевдо-случайное целое число в заданном диапазоне [a..b).

        Parameters
        ----------
        range_values: int | tuple[int, int]=(1 << 32)
            диапазон [a..b), если указано только один аргумент, то [0..b), если аргументов нет, то [0, 4294967296)

        Returns
        -------
        int
            псевдо-случайное целое число в заданном диапазоне

        Raises
        ------
        TypeError
            если переданные аргументы не являются ini или tuple[int,int]
        NotImplementedError
            если заданные диапазон для генереции больше, чем 2^32
        """

        left, right = 0, 0
        if type(range_values) == tuple:
            left, right = range_values
        elif type(range_values) == int:
            left, right = 0, range_values
        else:
            raise TypeError('Wrong type parameters.')
        if left > right:
            left, right = right, left
        rng = right - left
        if rng <= 1 << 32:
            threshold = (1 << 32) % rng
            while True:
                r = self.extract_number()
                if r >= threshold:
                    return left + (r % rng)
        else:
            raise NotImplementedError('To large range for 32-bit generator.')


    def __fillarr(self, arr: NDArray[Any], range_values: int | tuple[int, int]):
        if arr.ndim == 1:
            for i in range(len(arr)):
                arr[i] = self.randint(range_values)
        else:
            for i in arr:
                self.__fillarr(i, range_values)


    def randarrint(self, shape: int | tuple[int, ...], range_values: int | tuple[int, int]=(1 << 32)) -> NDArray[Any]:
        """
        Генерирует массив NDArray псевдо-случайных целых чисел заданной формы в заданном диапазоне [a..b).

        Parameters
        ----------
        shape:  int | tuple[int, ...]
            форма массива, размерность по всем измерениям
        range_values: int | tuple[int, int]=(1 << 32)
            диапазон [a..b), если указано только один аргумент, то [0..b), если аргументов нет, то [0, 4294967296)
        Returns
        -------
        NDArray
            массив псевдо-случайных целых чисел заданной формы в заданном диапазоне
        """

        arr = zeros(shape=(shape), dtype=int)
        self.__fillarr(arr, range_values)
        return arr


def B2D(vector_bits: NDArray[uint8]) -> uint8:
    """
    Перевод числа из двоичной вектор-строки длиной 8 символов в десятичный байт.

    Parameters
    ----------
    vector_bits: NDArray[uint8]
        двоичная вектор-строка длинной 8 символов

    Returns
    -------
    uint8
        десятичный байт
    """

    byte = uint8(0)
    for i in range(8):
        byte += vector_bits[i] * pow(2, i)
    return byte


def D2B(byte: uint8) -> NDArray[uint8]:
    """
    Перевод числа из десятичного байта в двоичную вектор-строку.

    Parameters
    ----------
    byte: uint8
        десятичный байт
    
    Returns
    -------
    NDArray[uint8]
        двоичная вектор-строка длинной 8 символов
    """
    
    vector_bits = zeros(8, dtype=uint8)
    for i in range(8):
        vector_bits[i] = byte % 2
        byte = byte // 2
    return vector_bits


def to_bit_vector(vector_bytes: NDArray[uint8]) -> NDArray[uint8]:
    """
    Преобразовать десятичную байтовую вектор-строку произвольной длинны (n) в двоичную вектор-строку длинны (n * 8).
    
    Parameters
    ----------
    vector_bytes: NDArray[uint8]
        десятичная байтовая вектор-строка
    
    Returns
    -------
    NDArray[uint8]
        двоичная вектор-строка
    """

    vector_bytes_len = len(vector_bytes)
    vector_bits = zeros(vector_bytes_len * 8, dtype=uint8)

    for i in range(vector_bytes_len):
        b = D2B(vector_bytes[i])
        for j in range(8):
            vector_bits[i * 8 + j] = b[j]

    return vector_bits


def from_bit_vector(vector_bits: NDArray[uint8]) -> NDArray[uint8]:
    """
    Преобразовать двоичную вектор-строку длинной (n) в десятичную байтовую вектор-строку длинной (n // 8).

    Parameters
    ----------
    vector_bits: NDArray[uint8]
        двоичная вектор-строка

    Returns
    -------
    NDArray[uint8]
        десятичная вектор-строка
    """

    vector_bits_len = len(vector_bits)
    vector_bytes = zeros(vector_bits_len // 8, dtype=uint8)
    vector_bytes_len = len(vector_bytes)
    b = zeros(8, dtype=uint8)

    for i in range(vector_bytes_len):
        b.fill(0)
        for j in range(8):
            b[j] = vector_bits[i * 8 + j]
        vector_bytes[i] = B2D(b)

    return vector_bytes


def bytes2chars(vector_bytes: NDArray[uint8]) -> str:
    """
    Преобразовать вектор-строку десятичных байтов в строку, соответствующих им, символов.
    
    Parameters
    ----------
    vector_bytes: NDArray[uint8]
        вектор-строка байтов

    Returns
    -------
    str
        строка символов
    """

    vector_chars = ''.join([chr(b) for b in vector_bytes])
    return vector_chars


def chars2bytes(vector_chars: str) -> NDArray[uint8]:
    """
    Преобразовать вектор-строку символов в вектор-строку, соответствующих им, десятичных байтов.

    Parameters
    ----------
    vector_chars: str
        вектор-строка символов

    Returns
    -------
    NDArray[int]
        вектор-строка байтов
    """
    
    return array([ord(ch) for ch in list(vector_chars)], dtype=uint8)


def key_pairs_gen(primary_key: int, count_key_pairs: int) -> NDArray[uint8]:
    """
    Генерация вектора пар ключей на основании первичного ключа.

    Parameters
    ----------
    primary_key: int
        первичный ключ
    count_key_pairs: int
        количество генерируемых пар ключей
    Returns
    -------
    NDArray[uint8]
        вектор-строка, содержащая count_key_pairs пар ключей

    """

    key_pairs = zeros(2 * count_key_pairs, dtype=uint16)
    key_pairs[0] = primary_key
    for i in range(1, 2 * count_key_pairs):
        key_pairs[i] = int(str(key_pairs[i - 1] ** 2)[:3])
        if key_pairs[i] > 255:
            key_pairs[i] = int(str(key_pairs[i])[0:2])
    return key_pairs.astype(uint8)


def step(byte: NDArray[uint8], scale: int) -> int:
    """
    Генерация псевдослучайного интервала на основе количества единиц
    в двоичном представлении номера последнего модифицированного байта 
    изображения умноженного на коэффициент масштабирования
    
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


def img_arr_surfs_to_one_arr(img_arr: NDArray[uint8] | None) -> tuple[NDArray[uint8] | None, int, int, int]:

    if img_arr is None: return None, 0, 0, 0

    count_dim = len(img_arr.shape)
    
    if count_dim == 3:
        surf_red = img_arr[:, :, 0]
        surf_green = img_arr[:, :, 1]
        surf_blue = img_arr[:, :, 2]
        one_arr = hstack((surf_red, surf_green, surf_blue))
        return one_arr, surf_red.shape[1], surf_green.shape[1], surf_blue.shape[1]
    elif count_dim == 2:
        one_arr = img_arr[:, :]
        return one_arr, one_arr.shape[1], 0, 0


    else:
        return None, 0, 0, 0


def one_arr_to_img_arr_surfs(one_arr: NDArray[uint8], red_width: int=0, green_width: int=0, blue_width: int=0) -> NDArray[uint8] | None:

    if red_width == 0 or green_width == 0 or blue_width == 0:
        return one_arr
    else:
        surf_red = one_arr[:, :blue_width]
        surf_green = one_arr[:, blue_width: blue_width + green_width]
        surf_blue = one_arr[:, blue_width + green_width: blue_width + green_width + red_width]
        img_arr = dstack((surf_red, surf_green, surf_blue))
        return img_arr




def img_arr_to_vect(img_arr: NDArray[uint8] | None) -> tuple[NDArray[uint8] | None, int, int, int]:
    """
    Преобразование массива, содержащего значения цветовых компонент (для цветного)
    или яркости (для монохромного) для каждого пикселя изображения, в вектор-строку байт.

    Parameters
    ----------
    img_arr : NDArray[uint8] | None
        массив, содержащий значения цветовых компонент или яркости для каждого пикселя изображения
    Returns
    -------
    tuple[NDArray[uint8] | None, int, int]
        img_vect : NDArray[uint8] | None
            вектор-строка байт, полученная из входного массива
        cover_len : int
            длина вектор-строки байт
        count_lines : int
            количество строк, которые были в исходном массиве
        count_dim : int
            количество измерений (2 - монохромное изображение, 3 - цветное изображение)
    """

    if img_arr is None: return None, 0, 0, 0
    count_lines = 0
    count_dim = len(img_arr.shape)
    if count_dim == 3:
        # получаем цветовые составляющие изображения
        surf_red = concatenate(img_arr[:, :, 0])
        surf_green = concatenate(img_arr[:, :, 1])
        surf_blue = concatenate(img_arr[:, :, 2])
        # собираем все цветовые составляющие в одну вектор-строку байт
        img_vect = concatenate([surf_red, surf_green, surf_blue])
        count_lines = len(img_arr[:, :, 0])
    elif count_dim == 2:
        # монохромное изображение
        img_vect = concatenate(img_arr)
        count_lines = len(img_arr[:, 0])
    else:
        return None, 0, 0, 0
    
    return img_vect, len(img_vect), count_lines, count_dim


def img_vect_to_arr(img_vect: NDArray[uint8], count_lines: int, count_dim: int) -> NDArray[uint8] | None:
    """
    Восстановление массива, содержащего значения цветовых компонент (для цветного)
    или яркости (для монохромного) для каждого пикселя изображения,
    из вектор-строки байт.

    Parameters
    ----------
    img_vect : NDArray[uint8]
        вектор-строка байт
    count_lines : int
        количество строк, которые были в исходном массиве
    Returns
    -------
    NDArray[uint8]
        восстановленный массив
    """

    img_arr = None
    if count_dim == 3:
        surf_red, surf_green, surf_blue = array_split(img_vect, 3)
        surf_red = array_split(surf_red, count_lines)
        surf_green = array_split(surf_green, count_lines)
        surf_blue = array_split(surf_blue, count_lines)
        img_arr = dstack((surf_red, surf_green, surf_blue))
    elif count_dim == 2:
        img_arr = array(array_split(img_vect, count_lines))
    
    return img_arr


if __name__ == '__main__':
    sys.exit()
