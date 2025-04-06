# блочное встраивание в НЗБ пикселей изображения сообщения для большей стойкости в случае изменения НЗБ изображения
# стегосистемы с использованием широкополосных сигналов

import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray, set_printoptions, inf, uint8, fromfile, hstack, copy, dstack, empty
from utils import D2B, B2D
from math import ceil, floor
from random import randrange

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)


def block_LSB_embedding(C, M):
    # цветовые компоненты изображения
    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]

    C_ = hstack((Cb, Cg, Cr))

    Nm = len(M) * 8

    X, Y = C_.shape

    Xa = ceil(Nm / Y)  # количество бит встраиваемых в строку
    print('Xa: ', Xa)

    Mb = empty(Nm, dtype=uint8)
    for i in range(len(M)):
        b = D2B(M[i])
        for j in range(8):
            Mb[i * 8 + j] = b[j]

    S_ = copy(C_)

    for i in range(Xa):
        r1 = i * floor(X / Xa)  # начало блока
        r2 = (i + 1) * floor(X / Xa)  # конец блока
        for y in range(Y):
            if y + i * Y >= Nm:  # если сообщение закончилось, то выходим
                break
            d = S_[r1:r2, y]  # вырезаем блок
            b = 0
            for x in range(len(d)):  # определяем четность блока
                P = D2B(d[x])
                LSB = P[0]
                b = b ^ LSB
            if b != Mb[y + i * Y]:  # если четность блока не совпадает с четностью бит сообщения
                n = ceil(randrange(len(d)))  # то меняем произвольный бит в блоке
                if d[n] % 2 == 0:
                    d[n] += 1
                else:
                    d[n] -= 1
                S_[r1:r2, y] = d  # помещаем измененный блок обратно

    S_[S_.shape[0] - 1, S_.shape[1] - 1] = Xa

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

    S = block_LSB_embedding(C, M)

    with Image.fromarray(S) as F:
        F.save(stg)


    fig = plt.figure('block LSB embedding')
    ax = fig.subplots(2, 4)

    ax[0, 0].set_title('source image')
    ax[0, 0].imshow(C)
    ax[1, 0].set_title('stego image')
    ax[1, 0].imshow(S)

    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]

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


