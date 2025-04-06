# PRI - pseudorandom interval
# встраивание в НЗБ пикселей изображения сообщения с псевдослучайным расстоянием между изменяемыми пикселями
# псевдослучайное распределение сообщения по всему изображению

from PIL import Image
from numpy import asarray, uint8, set_printoptions, inf, hstack, concatenate, empty, dstack, fromfile, array, copy
import matplotlib.pyplot as plt
from utils import D2B, B2D, step

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)

# встраивание в изображение C сообщения M, начиная с пикселя Ms, ограниченного в конце
# меткой Me, с масштабированием шага встраивания в k-раз
def LSB_PRI_embedding(C, M, Ms, Me, k):
    Nm = len(M)  # длинна встраиваемого сообщения

    # цветовые компоненты изображения
    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]

    Me = array([ord(i) for i in list(Me)])

    Me = concatenate((M, Me))  # добавляем метку конца сообщения к сообщению

    # соединяем матрицы цветов бок в бок друг за другом
    C_ = hstack((Cb, Cg, Cr))
    Cv = concatenate(C_)  # выстраиваем получившуюся матрицу в вектор построчно

    z = Ms
    Sv = copy(Cv)
    for i in range(len(Me)):
        b = D2B(Me[i])
        for j in range(8):
            P = D2B(Cv[z])
            P[0] = b[j]
            Sv[z] = B2D(P)
            z += step(D2B(z), k)

    S_ = Sv.reshape(C_.shape)
    Sb = S_[:, :Cb.shape[1]]
    Sg = S_[:, Cb.shape[1]: Cb.shape[1] + Cg.shape[1]]
    Sr = S_[:, Cb.shape[1] + Cg.shape[1]: Cb.shape[1] + Cg.shape[1] + Cr.shape[1]]

    S = dstack((Sr, Sg, Sb))

    return S


if __name__ == '__main__':
    img = 'assets/test1.bmp'  # изображение, в которое встраиваем сообщение
    stg = 'assets/stego.bmp'  # выходное изображение со встроенным сообщением
    msg = 'assets/message2.txt'  # сообщение, которое необходимо скрыть
    # ограничивающие метки начала и конца сообщения в файле изображения
    Ms = 54  # стартовая позиция с которой начинается встраивание (номер пикселя)
    Me = 'KiHeu,6'  # маркер окончания встроенного сообщения
    k = 10  # масштабирующий коэффициент для шага псевдослучайного интервала

    with Image.open(img) as F:
        C = asarray(F, dtype=uint8)  # изображение в которое будем встраивать сообщение

    with open(msg) as F:
        M = fromfile(F, dtype=uint8)


    S = LSB_PRI_embedding(C, M, Ms, Me, k)


    with Image.fromarray(S) as F:
        F.save(stg)


    # попытка визуальной атаки

    # выделяем цветовые компоненты исходного изображения
    Cr = C[:, :, 0]
    Cg = C[:, :, 1]
    Cb = C[:, :, 2]
    # выделяем цветовые компоненты исходного изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    # составляем массивы из младших бит пикселей цветовых составляющих
    Cr_v = Cr % 2  # исходные цветовые составляющие
    Cg_v = Cg % 2
    Cb_v = Cb % 2
    Sr_v = Sr % 2  # после встраивания сообщения
    Sg_v = Sg % 2
    Sb_v = Sb % 2

    # отображаем исходную картинку и со встроенным сообщением,
    # а также их цветовые составляющие после визуальной атаки
    fig = plt.figure('LSB pseudo random interval embedding')
    ax = fig.subplots(3, 4)

    ax[0, 0].set_title("source image")
    ax[0, 0].imshow(C)

    ax[0, 1].set_title("R source color\n visual attack")
    ax[0, 1].imshow(Cr_v)
    ax[0, 2].set_title("G source color\n visual attack")
    ax[0, 2].imshow(Cg_v)
    ax[0, 3].set_title("B source color\n visual attack")
    ax[0, 3].imshow(Cb_v)

    ax[1, 0].set_title("stego image")
    ax[1, 0].imshow(S)

    ax[1, 1].set_title("R stego color\n visual attack")
    ax[1, 1].imshow(Sr_v)
    ax[1, 2].set_title("G stego color\n visual attack")
    ax[1, 2].imshow(Sg_v)
    ax[1, 3].set_title("B stego color\n visual attack")
    ax[1, 3].imshow(Sb_v)


    # подсвечиваем пиксели, которые были изменены другим цветом

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
    ax[2, 1].set_title("R changed bits")
    ax[2, 1].imshow(C_Ch)

    G_ch, R_ch, B_ch = highlightchanges(Sg, Cg, Cr, Cb)
    C_Ch = dstack((R_ch, G_ch, B_ch))
    ax[2, 2].set_title("R changed bits")
    ax[2, 2].imshow(C_Ch)

    B_ch, G_ch, R_ch = highlightchanges(Sb, Cb, Cg, Cr)
    C_Ch = dstack((R_ch, G_ch, B_ch))
    ax[2, 3].set_title("R changed bits")
    ax[2, 3].imshow(C_Ch)

    plt.show()


