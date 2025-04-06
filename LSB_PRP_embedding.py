# PRP - pseudorandom permutation
# встраивание в НЗБ пикселей изображения сообщения с псевдослучайной перестановкой
# встраиваемых бит сообщения

from PIL import Image
from numpy import asarray, set_printoptions, inf, uint8, fromfile, copy, empty, dstack, hstack
from utils import keypairgen, D2B, B2D
from math import floor
import matplotlib.pyplot as plt

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)


def LSB_PRP_embedding(C, M, Ko, K):
    # цветовые компоненты изображения
    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]

    C_ = hstack((Cb, Cg, Cr))

    X, Y = C_.shape
    KT = keypairgen(Ko, K)  # генерируем пары ключей
    Nm = len(M) * 8  # длинна сообщения в битах

    # print(len(M))

    # переводим сообщение в массив двоичных бит
    Mb = empty(Nm, dtype=uint8)
    for i in range(len(M)):
        b = D2B(M[i])
        for j in range(8):
            Mb[i * 8 + j] = b[j]

    # xc = empty(Nm, dtype=uint8)
    # yc = empty(Nm, dtype=uint8)

    # print(Mb)
    # встраиваем сообщение
    S_ = copy(C_)
    for i in range(Nm):
        x = floor(i / Y)
        y = i % Y
        for s in range(K):
            x = (x + B2D(D2B(KT[2 * s - 1]) ^ D2B(y))) % X
            y = (y + B2D(D2B(KT[2 * s]) ^ D2B(x))) % Y
        P = D2B(S_[x, y])
        P[0] = Mb[i]
        S_[x, y] = B2D(P)

        # xc[i] = x
        # yc[i] = y

    # for i in range(len(xc)):
    #     print(xc[i], yc[i], end=' - ')
    #     for j in range(i + 1, len(xc)):
    #         if xc[i] == xc[j] and yc[i] == yc[j]:
    #             print(j, end=' ')
    #     print()


    Sb = S_[:, :Cb.shape[1]]
    Sg = S_[:, Cb.shape[1]: Cb.shape[1] + Cg.shape[1]]
    Sr = S_[:, Cb.shape[1] + Cg.shape[1]: Cb.shape[1] + Cg.shape[1] + Cr.shape[1]]

    S = dstack((Sr, Sg, Sb))

    return S


if __name__ == '__main__':
    img = 'assets/test1.bmp'  # изображение, в которое встраиваем сообщение
    stg = 'assets/stego.bmp'  # выходное изображение со встроенным сообщением
    msg = 'assets/message2.txt'  # сообщение, которое необходимо скрыть

    with Image.open(img) as F:
        C = asarray(F, dtype=uint8)  # изображение в которое будем встраивать сообщение

    with open(msg) as F:
        M = fromfile(F, dtype=uint8)  # секретное сообщение

    Ko = 125  # ключ
    K = 10  # количество генерируемых пар ключей
    # встраиваем в изображение сообщение
    S = LSB_PRP_embedding(C, M, Ko, K)


    # сохраняем результат в файл изображения
    with Image.fromarray(S) as F:
        F.save(stg)


    # отображаем исходную картинку и со встроенным сообщением,
    fig = plt.figure('LSB pseudo random permutation embedding')
    ax = fig.subplots(2, 4)

    ax[0, 0].set_title("source image")
    ax[0, 0].imshow(C)
    ax[1, 0].set_title("stego image")
    ax[1, 0].imshow(S)

    # выделяем цветовые компоненты исходного изображения
    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]
    # выделяем цветовые компоненты измененного изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    def highlightchanges(S, C1, C2, C3):
        ch_px = 0
        Ch1 = empty(C1.shape)
        Ch2 = empty(C2.shape)
        Ch3 = empty(C3.shape)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if C1[i, j] != S[i, j]:
                    ch_px += 1
                Ch1[i, j] = 0 if C1[i, j] != S[i, j] else C1[i, j] / 255
                Ch2[i, j] = 1 if C1[i, j] != S[i, j] else 0
                Ch3[i, j] = 1 if C1[i, j] != S[i, j] else 0
        print('changed pixels: ', ch_px)
        return Ch1, Ch2, Ch3

    R_ch, G_ch, B_ch = highlightchanges(Sr, Cr, Cg, Cb)
    C_Ch = dstack((R_ch, G_ch, B_ch))
    ax[1, 1].set_title("R changed bits")
    ax[1, 1].imshow(C_Ch)

    G_ch, R_ch, B_ch = highlightchanges(Sg, Cg, Cr, Cb)
    C_Ch = dstack((R_ch, G_ch, B_ch))
    ax[1, 2].set_title("R changed bits")
    ax[1, 2].imshow(C_Ch)

    B_ch, G_ch, R_ch = highlightchanges(Sb, Cb, Cg, Cr)
    C_Ch = dstack((R_ch, G_ch, B_ch))
    ax[1, 3].set_title("R changed bits")
    ax[1, 3].imshow(C_Ch)

    plt.show()

