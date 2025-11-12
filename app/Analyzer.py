import sys
from typing import Any
from abc import ABC, abstractmethod

from PIL import Image
from numpy import asarray, uint8
from numpy.typing import NDArray


class Analyzer(ABC):
    """
    Абстрактный класс для стегоанализа файла.
    """

    analysing_object: NDArray[uint8] | None = None
    result_object: NDArray[uint8] | None = None
    params: dict[str, Any] | None = None


    def set_params(self, **params: dict[str, Any]):
        """
        Устанавливает параметры стегоанализа файла.

        Parameters
        ----------
        **params: dict[str, Any]
            словарь с параметрами.
        """

        self.params = params

    def load_analysing_file(self, analysing_file_path: str):
        """
        Загружает анализуруемый файл и помещает в свойство stego_object.

        Parameters
        ----------
        stego_file_path: str
            имя/путь к стеганограмме
        """

        with Image.open(analysing_file_path) as F:
            self.analysing_object = asarray(F, dtype=uint8)


    @abstractmethod
    def analysing(self):
        """
        Абстрактный метод, в котором реализуется алгоритм стегоанализа файла классами-потомками.
        """

        pass

    
    def save_result_file(self, result_file_path: str):
        """
        Сохраняет полученные результаты стегоанализа из свойства result_object в файл.

        Parameters
        ----------
        result_file_path: str
            имя/путь к файлу с результатами стегоанализа
        """

        if self.result_object is not None:
            with Image.fromarray(self.result_object) as F:
                F.save(result_file_path)


    def process_one_file(
            self,
            analysing_file_path: str,
            result_file_path: str,
            **params: dict[str, Any]
    ):
        """
        Произвести стегоанализ одного файла.

        Parameters
        ----------
        analysing_file_path: str
            имя/путь к файлу стеганограммы
        result_file_path: str
            путь к файлу вложения (только директория)
        **params: dict[str, Any]
            параметры стегоаналитического алгоритма, зависят от используемого алгоритма.
        """

        self.set_params(**params)
        self.load_analysing_file(analysing_file_path)
        self.analysing()
        if self.result_object is None: return
        self.save_result_file(result_file_path)


if __name__ == '__main__':
    sys.exit()
