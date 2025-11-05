import sys

import matplotlib.pyplot as plt
from numpy import expand_dims, frombuffer, uint8
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .Analyzer import Analyzer


# значения по умолчанию параметров уникальных для алгоритма
default_num_surface = 0


class SGA_visual(Analyzer):
    """
    Реализация визуальной атаки на стеганограмму в заданной битовой влоскости (visual).
    Получает из свойства родителя params параметр работы:
    {'num_surface': 0}
        номер битовой плоскости 0 - наименее, 7 - наиболее значащие биты
    """

    def analysing(self):
        if self.analysing_object is None: return

        # получаем параметры работы алгоритма
        num_surface = (self.params or {}).get('num_surface', default_num_surface)

        if len(self.analysing_object.shape) < 3:
            self.analysing_object = expand_dims(self.analysing_object, axis=2)

        fig = plt.figure(figsize=(8, 3), dpi=1200)  # type: ignore
        fig.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2, wspace=0, hspace=0)
        fig.subplots_adjust(wspace=0.5)
        ax = fig.subplots(1, self.analysing_object.shape[2])  # type: ignore

        for i in range(self.analysing_object.shape[2]):
            # выбираем цветовую составляющую
            color_plane = self.analysing_object[:, :, i]
            # делаем срез цветовой составляющей, получаем битовую плоскость
            bit_plane = (color_plane >> num_surface) & 1
            axis = ax if self.analysing_object.shape[2] == 1 else ax[i]
            axis.set_title(f'plane {i}')  # type: ignore
            axis.imshow(bit_plane, cmap='gray')  # type: ignore

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = canvas.tostring_argb()
        argb = frombuffer(buf, dtype=uint8).reshape(h, w, 4)  # type: ignore
        self.result_object = argb[:, :, 1:]
        plt.close(fig)  #type: ignore


if __name__ == '__main__':
    sys.exit()
