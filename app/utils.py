import sys

from numpy import empty, array, uint8, uint16, ndarray


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
    
    vector_bits = empty(8, dtype=uint8)
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
    vector_bits = empty(vector_bytes_len * 8, dtype=uint8)

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
    vector_bytes = empty(vector_bits_len // 8, dtype=uint8)
    vector_bytes_len = len(vector_bytes)
    b = empty(8, dtype=uint8)

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
    Ks = empty(2 * K, dtype=uint16)
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
