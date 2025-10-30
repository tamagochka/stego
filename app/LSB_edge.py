import sys

from numpy import dstack

from .Embedder import Embedder
from .Extractor import Extractor
from .utils import edge_detect


class LSB_edge_embedding(Embedder):
    """
    (edge)
    """

    def embedding(self):
        
        if self.cover_object is None: return

        # применяем оператор Прюитт
        edged_surfs = edge_detect(self.cover_object)
        if edged_surfs is None: return

        
        




        
        # собираем цветовые плоскости изображения (если они есть) в один массив
        if len(edged_surfs) > 1:
            self.stego_object = dstack(edged_surfs)
        else:
            self.stego_object = edged_surfs[0]










class LSB_edge_extracting(Extractor):
    """
    (edge)
    """

    def extracting(self):
        
        print('Hello!')


if __name__ == "__main__":
    sys.exit()
    