import sys, logging, argparse
from argparse import Namespace
from dataclasses import dataclass


from .config import AppConfig
from .LSB import LSB_embedding, LSB_extracting
from .LSB_PRI import LSB_PRI_embedding, LSB_PRI_extracting
from .LSB_PRP import LSB_PRP_embedding, LSB_PRP_extracting
from .LSB_block import LSB_block_embedding, LSB_block_extracting
from .steganalysing import visual_attack


@dataclass
class App(object):

    config: AppConfig = None
    args: Namespace = None
    parser: argparse.ArgumentParser = None
    subparsers: dict[str, argparse.ArgumentParser] = None
    available_args_for_embedding: list[str] = None
    available_args_for_extracting: list[str] = None
    available_args_for_analysing: list[str] = None

    def embedding(self):
        # если никакие параметры не переданы, то выводим справку и выходим
        if all(((key in self.available_args_for_embedding) == (val is None)) for key, val in vars(self.args).items()):
            self.subparsers['embedding'].print_help()
            return
        cover_file_path = self.config.get_covers_file_path(self.args.cover) if self.args.cover else None
        stego_file_path = self.config.get_stegos_file_path(self.args.stego) if self.args.stego else None
        message_file_path = self.config.get_messages_file_path(self.args.message) if self.args.message else None
        algorithm = self.args.algorithm
        # преобразуем строку параметров в словарь
        params = None
        if self.args.params:
            params = eval(f'dict({self.args.params})')

        match algorithm:
            case 'lsb':
                LSB_embedding(
                    cover_file_path,
                    stego_file_path,
                    message_file_path,
                    **params if params else {}  # распаковываем параметры из словаря, если они были переданы
                )
            case 'pri':
                LSB_PRI_embedding(
                    cover_file_path,
                    stego_file_path,
                    message_file_path,
                    **params if params else {}
                )
            case 'prp':
                LSB_PRP_embedding(
                    cover_file_path,
                    stego_file_path,
                    message_file_path,
                    **params if params else {}
                )
            case 'block':
                LSB_block_embedding(
                    cover_file_path,
                    stego_file_path,
                    message_file_path,
                    **params if params else {}
                )


    def extracting(self):
        # если никакие параметры не переданы, то выводим справку и выходим
        if all(((key in self.available_args_for_extracting) == (val is None)) for key, val in vars(self.args).items()):
            self.subparsers['extracting'].print_help()
            return
        stego_file_path = self.config.get_stegos_file_path(self.args.stego)
        extract_file_path = self.config.get_extracts_file_path()
        algorithm = self.args.algorithm
        params = None
        if self.args.params:
            params = eval(f'dict({self.args.params})')

        match algorithm:
            case 'lsb':
                LSB_extracting(
                    stego_file_path,
                    extract_file_path,
                    **params if params else {}
                )
            case 'pri':
                LSB_PRI_extracting(
                    stego_file_path,
                    extract_file_path,
                    **params if params else {}
                )
            case 'prp':
                LSB_PRP_extracting(
                    stego_file_path,
                    extract_file_path,
                    **params if params else {}
                )
            case 'block':
                LSB_block_extracting(
                    stego_file_path,
                    extract_file_path,
                    **params if params else {}
                )


    def analysing(self):
        # если никакие параметры не переданы, то выводим справку и выходим
        if all(((key in self.available_args_for_analysing) == (val is None)) for key, val in vars(self.args).items()):
            self.subparsers['analysing'].print_help()
            return
        stego_file_path = self.config.get_stegos_file_path(self.args.stego)
        result_file_path = self.config.get_analysis_file_path(self.args.result) if self.args.result else None
        algorithm = self.args.algorithm
        params = None
        if self.args.params:
            params = eval(f'dict({self.args.params})')

        match algorithm:
            case 'visual':
                visual_attack(
                    stego_file_path,
                    result_file_path,
                    **params if params else {}
                )


    def parser_init(self) -> Namespace:

        self.parser = argparse.ArgumentParser(description='Стеганографическое погружение/извлечение/анализ информации.')
        self.parser.add_argument('-cfg', '--config', type=str, help='Путь к файлу конфигурации в формате toml.', default=None)
        subparsers = self.parser.add_subparsers()
        self.available_args_for_embedding = ['config']
        self.available_args_for_extracting = ['config']
        self.available_args_for_analysing = ['config']


        parser_embedding = subparsers.add_parser('embedding', help='Погружение информации.', formatter_class=argparse.RawTextHelpFormatter)
        self.subparsers.update({'embedding': parser_embedding})
        parser_embedding.add_argument('-a', '--algorithm', type=str,
                                help='Алгоритм, используемый для погружения вложения в покрывающий объект.\n' \
                                    'Поддерживаемые алгоритмы:\n' \
                                            '\tlsb - погружение информации в плоскость наименее значащих бит (НЗБ) с непрерывным заполнением.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tstart_label: str - метка начала места погружения;\n' \
                                                    '\t\t\tend_label: srt - метка конца места погружения;\n' \
                                                    '\t\t\tfill_rest: bool - заполнять незаполненную часть покрывающего объекта случайными битами.\n' \
                                            '\tpri - погружение информации в НЗБ с псевдослучайным интервалом.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tstart_label: int - место начала погружения;\n' \
                                                    '\t\t\tend_label: str - метка конца места погружения;\n' \
                                                    '\t\t\tkey: int - ключ, задающий масштабирование шага встраивания;\n' \
                                                    '\t\t\tfill_rest: bool - заполнять незаполненную часть покрывающего объекта случайными битами.\n' \
                                            '\tprp - погружение информации в НЗБ с псевдослучайной перестановкой бит вложения.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tprimary_key: int - первичный ключ, используемый для генерации перестановок;\n' \
                                                    '\t\t\tcount_key_pairs: int - количество генерируемых пар ключей перестановок;\n' \
                                                    '\t\t\tend_label: str - метка конца места погружения.\n')
        self.available_args_for_embedding.append('algorithm')
        parser_embedding.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для погружения.')
        self.available_args_for_embedding.append('params')
        parser_embedding.add_argument('-c', '--cover', type=str, help='Путь к файлу - покрывающему объекту (контейнеру),' \
                                'в который осуществляется погружение (встраивание) вложения (скрываемой информации). ' \
                                'Поддерживаемые типы файлов: bmp.')
        self.available_args_for_embedding.append('cover')
        parser_embedding.add_argument('-m', '--message', type=str, help='Путь к файлу вложению (скрываемой информации),' \
                                'которое погружается (встраивается) в покрывающий объект (контейнер). ' \
                                'Поддерживаются любые типы файлов.')
        self.available_args_for_embedding.append('message')
        parser_embedding.add_argument('-s', '--stego', type=str, help='Путь к файлу - стеганограмме результату погруженния вложения в покрывающий объект.')
        self.available_args_for_embedding.append('stego')
        parser_embedding.set_defaults(func=self.embedding)


        parser_extracting = subparsers.add_parser('extracting', help='Извлечение информации.', formatter_class=argparse.RawTextHelpFormatter)
        self.subparsers.update({'extracting': parser_extracting})
        parser_extracting.add_argument('-a', '--algorithm', type=str,
                                    help='Алгоритм, который был использован для погружения.\n' \
                                        'Поддерживаемые алгоритмы:\n' \
                                            '\tlsb - извлечение информации из плоскости наименее значащих бит (НЗБ) с непрерывным заполнением.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tstart_label: str - метка начала места погружения;\n' \
                                                    '\t\t\tend_label: srt - метка конца места погружения.\n' \
                                            '\tpri - извлечение информации из НЗБ с псевдослучайным интервалом.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tstart_label: int - место начала погружения;\n' \
                                                    '\t\t\tend_label: str - метка конца места погружения;\n' \
                                                    '\t\t\tkey: int - ключ, задающий масштабирование шага встраивания.\n' \
                                            '\tprp - извлечение информации из НЗБ с псевдослучайной перестановкой бит вложения.\n' \
                                                '\t\tдоступные параметры алгоритма:\n' \
                                                    '\t\t\tprimary_key: int - первичный ключ, используемый для генерации перестановок;\n' \
                                                    '\t\t\tcount_key_pairs: int - количество генерируемых пар ключей перестановок;\n' \
                                                    '\t\t\tend_label:str - метка конца места погружения.')
        self.available_args_for_extracting.append('algorithm')
        parser_extracting.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для извлечения.')
        self.available_args_for_extracting.append('params')
        parser_extracting.add_argument('-s', '--stego', type=str, help='Стеганограмма из которого осуществляется извлечение вложения (скрытой в ней информации). ' \
                                'Поддерживаемые типы файлов: bmp.')
        self.available_args_for_extracting.append('stego')
        parser_extracting.set_defaults(func=self.extracting)


        parser_analysing = subparsers.add_parser('analysing', help='Стеганографический анализ.', formatter_class=argparse.RawTextHelpFormatter)
        self.subparsers.update({'analysing': parser_analysing})
        parser_analysing.add_argument('-a', '--algorithm', type=str, help='Алгоритм, используемый для анализа. ' \
                                'Поддерживаемые алгоритмы стегоанализа: visual.')
        self.available_args_for_analysing.append('algorithm')
        parser_analysing.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для анализа.')
        self.available_args_for_analysing.append('params')
        parser_analysing.add_argument('-s', '--stego', type=str, help='Файл, который будет подвергнут стеганографическому анализу (предполагаемая стеганограмма). ' \
                                'Поддерживаемые типы файлов: bmp.')
        self.available_args_for_analysing.append('stego')
        parser_analysing.add_argument('-r', '--result', type=str, help='Файл с результатами анализа. Тип файла, зависит от используемого алгоритма.')
        self.available_args_for_analysing.append('result')
        parser_analysing.set_defaults(func=self.analysing)


    def __init__(self):
        self.subparsers = {}

        self.parser_init()
        self.args = self.parser.parse_args()

        self.config = AppConfig(config_file=self.args.config)
        self.config.apply_config()


    def run(self):
        
        if not hasattr(self.args, 'func'):
            self.parser.print_help()
            sys.exit()

        self.args.func()


if __name__ == '__main__':
    sys.exit()

