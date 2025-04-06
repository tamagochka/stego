# извлечение сообщения из НЗБ пикселей изображения после простого встраивания

from PIL import Image
from numpy import asarray, uint8, set_printoptions, inf, hstack, concatenate, array, empty, append, zeros, fromstring, repeat, where
from utils import D2B, B2D, alphagen, keygen

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)

from LSB import LSB_extracting, LSB_embedding

LSB_extracting()
LSB_embedding()

# извлечение из изображения S сообщения M, которое было закодировано алфавитом A и зашифровано
# ключем K, сообщение ограничено метками Ms и Me
def LSB_extracting(S, Ms, Me, K, A):
    Nk = len(K)  # длинна секретного ключа
    Na = len(A)  # длинна алфавита

    # цветовые компоненты изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    # соединяем матрицы цветов бок в бок друг за другом
    S_ = hstack((Sb, Sg, Sr))
    # выстраиваем получившуюся матрицу в вектор построчно
    Sv = concatenate(S_)

    # берем НЗБ каждого пикселя, объединяем их по 8 в байты,
    # преобразуем в десятичное число и собираем в массив
    sEe = empty(len(Sv) // 8, dtype=uint8)
    b = empty(8, dtype=uint8)
    for i in range(len(Sv) // 8):
        b.fill(0)
        for j in range(8):
            P = D2B(Sv[j + 8 * i])
            b[j] = P[0]
        sEe[i] = B2D(b)

    # преобразуем полученный массив чисел в символы
    E_ = ''.join([chr(e) for e in sEe])
    s = E_.find(Ms) + len(Ms)  # ищем начальную ограничивающую метку + ее длинна - начало сообщения
    e = E_.find(Me)  # конец сообщения
    Nm = e - s  # длинна встроенного сообщения
    E = sEe[s:e]

    # увеличиваем ключ на длинну сообщения путем его повторения (можно генерировать ПСП на основе ключа)
    K_ = keygen(K, Nm)

    # расшифровываем сообщение
    M = empty(Nm, dtype=uint8)
    for j in range(Nm):
        m = where(A == E[j])[0][0]  # ищем соответствие встроенного символа символу алфавита
        n = where(A == K_[j])[0][0]  # соответствие символа алфавита секретному ключу
        r = (Na + m - n) % Na  # позиция символа сообщения в алфавите который был закодирован
        M[j] = A[r]

    M = ''.join(chr(m) for m in M)

    return M


if __name__ == '__main__':
    stg = 'assets/stego.bmp'  # изображение, в которое встроено сообщение
    K = '@J|eKc-I98O'  # секретный ключ, которым зашифровывается сообщение
    Ms = 'n04@m0k'  # ограничивающие метки начала
    Me = 'KiHeu,6'  # и конца сообщения в файле изображения
    Na = 256  # длинна алфавита

    with Image.open(stg) as F:
        S = asarray(F, dtype=uint8)

    # алфавит источника сообщения, можно переставить символы по определенному закону для большей защищенности
    A = alphagen(Na)  # генерируем алфавит


    # извллекаем сообщение из изображения
    M = LSB_extracting(S, Ms, Me, K, A)


    print(M)
