from abc import ABC, abstractmethod


class Extractor(ABC):

    @abstractmethod
    def extracting(self):
        pass
