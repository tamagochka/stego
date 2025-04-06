# простое встраивание в НЗБ пикселей изображения сообщения

from PIL import Image
from numpy import asarray, uint8, set_printoptions, inf, hstack, concatenate, empty, dstack, fromfile, array, where
import matplotlib.pyplot as plt
from utils import D2B, B2D, alphagen, keygen
from random import random

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)


# встраивание в изображение C сообщения M закодированное в алфавите A и зашифрованное ключем K
# начало и конец сообщения ограничены метками Ms и Me
def LSB_embedding(C, M, Ms, Me, K, A):
    Nm = len(M)  # длинна встраиваемого сообщения
    Nk = len(K)  # длинна секретного ключа
    Na = len(A)  # длинна алфавита

    # цветовые компоненты изображения
    # Cr = C[:, :, 0]
    # Cg = C[:, :, 1]
    # Cb = C[:, :, 2]


    # увеличиваем ключ на длинну сообщения путем его повторения (можно генерировать ПСП на основе ключа)
    # K_ = keygen(K, Nm)

    # кодирование сообщения M в заданном алфавите A и применение к нему ключа шифрования K
    # на выходе получаем криптограмму E
    Nm = len(M)
    E = M
    # Na = len(A)
    # E = empty(Nm, dtype=uint8)
    # for j in range(Nm):
    #     m = where(A == M[j])[0][0]  # какому значению в алфавите соответствует символ сообщения
    #     n = where(A == K_[j])[0][0]  # какому значению в алфавите соответствует символ ключа
    #     r = (m + n) % Na  # символ алфавита, которым будет закодирован символ сообщения
    #     E[j] = A[r]

    # преобразуем метки начала/конца сообщения в ASCII коды и присоеденим их к сообщению
    Ms = array([ord(i) for i in list(Ms)])
    Me = array([ord(i) for i in list(Me)])
    sEe = concatenate((Ms, E, Me))

    # соединяем матрицы цветов бок в бок друг за другом
    # C_ = hstack((Cb, Cg, Cr))
    # выстраиваем получившуюся матрицу в вектор построчно
    # Cv = concatenate(C_)
    Cv = concatenate(C)


    # len(Cv) > 8 * len(sMe) must!
    Sv = empty(len(Cv), dtype=uint8)
    for j in range(len(sEe)):
        b = D2B(sEe[j])  # преобразуем байт сообщения в двоичный вид
        for i in range(8):
            P = D2B(Cv[i + 8 * j])  # преобразуем байт контейнера в двоичный вид
            P[0] = b[i]  # прячем бит сообщения в байте контейнера
            Sv[i + 8 * j] = B2D(P)  # преобразуем измененный байт контейнера обратно в десятичный вид

    # заполняем оставшуюся часть контейнера случайными битами
    for j in range(len(sEe) * 8, len(Cv)):
        P = D2B(Cv[j])
        P[0] = round(random())  # случайно 0 или 1
        Sv[j] = B2D(P)

    # S_ = Sv.reshape(C_.shape)  # преобразуем вектор в массив
    S_ = Sv.reshape(C.shape)
    # выделяем из массива цветовые компоненты
    # Sb = S_[:, :Cb.shape[1]]
    # Sg = S_[:, Cb.shape[1]: Cb.shape[1] + Cg.shape[1]]
    # Sr = S_[:, Cb.shape[1] + Cg.shape[1]: Cb.shape[1] + Cg.shape[1] + Cr.shape[1]]

    # собираем итоговый массив изображения
    # S = dstack((Sr, Sg, Sb))
    S = S_
    return S


if __name__ == '__main__':
    img = 'assets/butterfly.pgm'  # изображение, в которое встраиваем сообщение
    stg = 'assets/stego.bmp'  # выходное изображение со встроенным сообщением
    msg = 'assets/message2.txt'  # сообщение, которое необходимо скрыть
    K = '@J|eKc-I98O'  # секретный ключ, которым зашифровывается сообщение
    Ms = 'n04@m0k'  # ограничивающие метки начала
    Me = 'KiHeu,6'  # и конца сообщения в файле изображения
    Na = 256  # длинна алфавита

    with Image.open(img) as F:
        C = asarray(F, dtype=uint8)  # изображение в которое будем встраивать сообщение

    with open(msg) as F:
        M = fromfile(F, dtype=uint8)  # секретное сообщение

    # алфавит источника сообщения, можно переставить символы по определенному закону для большей защищенности
    A = alphagen(Na)  # генерируем алфавит


    # встраиваем в изображение сообщение
    S = LSB_embedding(C, M, Ms, Me, K, A)


    # сохраняем результат в файл изображения
    with Image.fromarray(S) as F:
        F.save(stg)


    # визуальная атака на встроенное сообщение

    # выделяем цветовые компоненты исходного изображения
    # Cr = C[:, :, 0]
    # Cg = C[:, :, 1]
    # Cb = C[:, :, 2]
    # выделяем цветовые компоненты исходного изображения
    # Sr = S[:, :, 0]
    # Sg = S[:, :, 1]
    # Sb = S[:, :, 2]

    # составляем массивы из младших бит пикселей цветовых составляющих
    # Cr_v = Cr % 2  # исходные цветовые составляющие
    # Cg_v = Cg % 2
    # Cb_v = Cb % 2
    # Sr_v = Sr % 2  # после встраивания сообщения
    # Sg_v = Sg % 2
    # Sb_v = Sb % 2




    C_v = C % 2
    S_v = S % 2



    # отображаем исходную картинку и со встроенным сообщением,
    # а также их цветовые составляющие после визуальной атаки
    fig = plt.figure()
    ax = fig.subplots(2, 2)

    ax[0, 0].set_title("source image")
    ax[0, 0].imshow(C, cmap='gray')

    ax[0, 1].set_title("visual attack")
    ax[0, 1].imshow(C_v, cmap='gray')
    # ax[0, 1].imshow(Cr_v)
    # ax[0, 2].set_title("G source color\n visual attack")
    # ax[0, 2].imshow(Cg_v)
    # ax[0, 3].set_title("B source color\n visual attack")
    # ax[0, 3].imshow(Cb_v)

    ax[1, 0].set_title("stego image")
    ax[1, 0].imshow(S, cmap='gray')

    ax[1, 1].set_title("visual attack")
    ax[1, 1].imshow(S_v, cmap='gray')
    # ax[1, 1].imshow(Sr_v)
    # ax[1, 2].set_title("G stego color\n visual attack")
    # ax[1, 2].imshow(Sg_v)
    # ax[1, 3].set_title("B stego color\n visual attack")
    # ax[1, 3].imshow(Sb_v)

    plt.show()

