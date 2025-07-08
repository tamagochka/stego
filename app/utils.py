import sys

from numpy import zeros, array, uint8, uint16, ndarray


class MersenneTwister(object):
    """
    Генератор псевдослучайных чисел на основе алгоритма "Вихрь Мерсена"
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


    def __init__(self, seed: int=0xDEADBEEF):
        self.MT = [0] * self.n
        self.index = 0
        self.seed(seed)


    def seed(self, seed: int=0xDEADBEEF):
        self.MT[0] = seed & 0xFFFFFFFF
        for i in range(1, self.n):
            self.MT[i] = (1812433253 * (self.MT[i - 1] ^ self.MT[i - 1] >> 30)) + i
            self.MT[i] &= 0xFFFFFFFF
        self.index = self.n


    def extract_number(self) -> int:
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
        return self.extract_number() / 4294967296.0  # нормирование к диапазону [0; 1)


    def randint(self, range_values: int | tuple[int, int]=(1 << 32)) -> int:
        left, right = range_values if type(range_values) == tuple else (0, range_values)
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


    def __fillarr(self, arr: ndarray, range_values: int | tuple[int, int]):
        if arr.ndim == 1:
            for i in range(len(arr)):
                arr[i] = self.randint(range_values)
        else:
            for i in arr:
                self.__fillarr(i, range_values)


    def randarrint(self, range: int | tuple[int, int], shape: int | tuple[int, ...]) -> ndarray:
        arr = zeros(shape=(shape), dtype=int)
        self.__fillarr(arr, range)
        return arr


def B2D(vector_bits: ndarray[uint8]) -> uint8:
    """
    Перевод числа из двоичной вектор-строки длиной 8 символов в десятичный байт

    Parameters
    ----------
        vector_bits: ndarray[uint8]
            двоичная вектор-строка длинной 8 символов

    Returns
    -------
        uint8
            десятичный байт
    """
    byte = 0
    for i in range(8):
        byte += vector_bits[i] * pow(2, i)
    return byte


def D2B(byte: uint8) -> ndarray[uint8]:
    """
    Перевод числа из десятичного байта в двоичную вектор-строку

    Parameters
    ----------
        byte: uint8
            десятичный байт
    
    Returns
    -------
        ndarray[uint8]
            двоичная вектор-строка длинной 8 символов
    """
    
    vector_bits = zeros(8, dtype=uint8)
    for i in range(8):
        vector_bits[i] = byte % 2
        byte = byte // 2
    return vector_bits


def to_bit_vector(vector_bytes: list[int]) -> ndarray[int]:
    """
    Преобразовать десятичную байтовую вектор-строку произвольной длинны (n) в двоичную вектор-строку длинны (n * 8)
    
    Parameters
    ----------
        vector_bytes: list[int]
            десятичная байтовая вектор-строка
    
    Returns
    -------
        ndarray[int]
            двоичная вектор-строка
    """

    vector_bytes_len = len(vector_bytes)
    vector_bits = zeros(vector_bytes_len * 8, dtype=uint8)

    for i in range(vector_bytes_len):
        b = D2B(vector_bytes[i])
        for j in range(8):
            vector_bits[i * 8 + j] = b[j]

    return vector_bits


def from_bit_vector(vector_bits: ndarray[uint8]) -> ndarray[uint8]:
    """
    Преобразовать двоичную вектор-строку длинной (n) в десятичную байтовую вектор-строку длинной (n // 8)

    Parameters
    ----------
        vector_bits: ndarray[uint8]
            двоичная вектор-строка

    Returns
    -------
        ndarrayp[uint8]
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


def bytes2chars(vector_bytes: ndarray[uint8]) -> str:
    """
    Преобразовать вектор-строку десятичных байтов в строку, соответствующих им, символов
    
    Parameters
    ----------
        vector_bytes: ndarray[uint8]
            вектор-строка байтов

    Returns
    -------
        str
            строка символов
    """

    vector_chars = ''.join([chr(b) for b in vector_bytes])
    return vector_chars


def chars2bytes(vector_chars: str) -> ndarray[uint8]:
    """
    Преобразовать вектор-строку символов в вектор-строку, соответствующих им, десятичных байтов.

    Parameters
    ----------
        vector_chars: str
            вектор-строка символов

    Returns
    -------
        ndarray[int]
            вектор-строка байтов
    """
    
    return array([ord(ch) for ch in list(vector_chars)], dtype=uint8)


# алфавит источника сообщения, можно переставить символы по определенному закону для большей защищенности
def alphagen(Na):
    A = array([i - 1 for i in range(0, Na)], dtype=uint8)  # значения элементов сместили на 1 влево
    A[0] = Na
    return A


# увеличиваем ключ на длинну сообщения путем его повторения (можно генерировать ПСП на основе ключа)
def keygen(K, Nm):
    # k = array(list(K))
    return array([ord(array(list(K))[i % len(K)]) for i in range(Nm)], dtype=uint8)


def keypairgen(Ko, K):
    Ks = zeros(2 * K, dtype=uint16)
    for i in range(0, 2 * K):
        if i == 0:
            Ks[i] = Ko
        if i > 0:
            Ks[i] = int(str(Ks[i - 1] ** 2)[:3])
        if Ks[i] > 255:
            Ks[i] = int(str(Ks[i])[0:2])
    return Ks.astype(uint8)


if __name__ == '__main__':
    sys.exit()
