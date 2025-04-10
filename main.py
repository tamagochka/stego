import logging
import argparse
from argparse import Namespace

from numpy import set_printoptions, inf

from config import AppConfig
from LSB import LSB_embedding, LSB_extracting
from LSB_PRI import LSB_PRI_embedding, LSB_PRI_extracting
from LSB_PRP import LSB_PRP_embedding
from steganalysing import visual_attack


set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)


def embedding(args: Namespace, app_config: AppConfig):
    cover_file_path = app_config.get_covers_file_path(args.cover) if args.cover else None
    stego_file_path = app_config.get_stegos_file_path(args.stego) if args.stego else None
    message_file_path = app_config.get_messages_file_path(args.message) if args.message else None
    algorithm = args.algorithm
    # преобразуем строку параметров в словарь
    params = None
    if args.params:
        params = eval(f'dict({args.params})')

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


def extracting(args: Namespace, app_config: AppConfig):
    stego_file_path = app_config.get_stegos_file_path(args.stego)
    extract_file_path = app_config.get_extracts_file_path()
    algorithm = args.algorithm
    params = None
    if args.params:
        params = eval(f'dict({args.params})')

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


def analysing(args: Namespace, app_config: AppConfig):
    stego_file_path = app_config.get_stegos_file_path(args.stego)
    result_file_path = app_config.get_analysis_file_path(args.result) if args.result else None
    algorithm = args.algorithm
    params = None
    if args.params:
        params = eval(f'dict({args.params})')

    match algorithm:
        case 'visual':
            visual_attack(
                stego_file_path,
                result_file_path,
                **params if params else {}
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Стеганографическое погружение/извлечение/анализ информации.')
    parser.add_argument('-cfg', '--config', type=str, help='Путь к файлу конфигурации в формате toml.', default=None)
    subparsers = parser.add_subparsers()

    parser_embedding = subparsers.add_parser('embedding', help='Погружение информации.', formatter_class=argparse.RawTextHelpFormatter)
    parser_embedding.add_argument('-a', '--algorithm', type=str,
                              help='Алгоритм, используемый для погружения вложения в покрывающий объект.\n' \
                                   'Поддерживаемые алгоритмы:\n' \
                                        '\tlsb - погружение информации в плоскость наименее значащих бит (НЗБ) с непрерывным заполнением.\n' \
                                            '\t\tдоступные параметры алгоритма:\n' \
                                                '\t\t\tstart_label:str - метка начала места погружения;\n' \
                                                '\t\t\tend_label:srt - метка конца места погружения;\n' \
                                                '\t\t\tfill_rest:bool - заполнять незаполненную часть покрывающего объекта случайными битами.\n' \
                                        '\tpri - погружение информации в НЗБ с псевдослучайным интервалом.\n' \
                                            '\t\tдоступные параметры алгоритма:\n' \
                                                '\t\t\tstart_label:int - место начала погружения;\n' \
                                                '\t\t\tend_label:str - метка конца места погружения;\n' \
                                                '\t\t\tkey:int - ключ, задающий масштабирование шага встраивания;' \
                                                '\t\t\tfill_rest - заполнять незаполненную часть покрывающего объекта случайными битами.')
    parser_embedding.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для погружения.')
    parser_embedding.add_argument('-c', '--cover', type=str, help='Путь к файлу - покрывающему объекту (контейнеру),\
                               в который осуществляется погружение (встраивание) вложения (скрываемой информации).\
                              Поддерживаемые типы файлов: bmp.')
    parser_embedding.add_argument('-m', '--message', type=str, help='Путь к файлу вложению (скрываемой информации),\
                              которое погружается (встраивается) в покрывающий объект (контейнер).\
                              Поддерживаются любые типы файлов.')
    parser_embedding.add_argument('-s', '--stego', type=str, help='Путь к файлу - стеганограмме результату погруженния вложения в покрывающий объект.')
    parser_embedding.set_defaults(func=embedding)


    parser_extracting = subparsers.add_parser('extracting', help='Извлечение информации.', formatter_class=argparse.RawTextHelpFormatter)
    parser_extracting.add_argument('-a', '--algorithm', type=str,
                                help='Алгоритм, который был использован для погружения.\n' \
                                     'Поддерживаемые алгоритмы:\n' \
                                        '\tlsb - извлечение информации из плоскости наименее значащих бит (НЗБ) с непрерывным заполнением.\n' \
                                            '\t\tдоступные параметры алгоритма:\n' \
                                                '\t\t\tstart_label:str - метка начала места погружения;\n' \
                                                '\t\t\tend_label:srt - метка конца места погружения.\n' \
                                        '\tpri - извлечение информации из НЗБ с псевдослучайным интервалом.\n' \
                                            '\t\tдоступные параметры алгоритма:\n' \
                                                '\t\t\tstart_label:int - место начала погружения;\n' \
                                                '\t\t\tend_label:str - метка конца места погружения;\n' \
                                                '\t\t\tkey:int - ключ, задающий масштабирование шага встраивания.')
    parser_extracting.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для извлечения.')
    parser_extracting.add_argument('-s', '--stego', type=str, help='Стеганограмма из которого осуществляется извлечение вложения (скрытой в ней информации).\
                                Поддерживаемые типы файлов: bmp.')
    parser_extracting.set_defaults(func=extracting)


    parser_analysing = subparsers.add_parser('analysing', help='Стеганографический анализ.', formatter_class=argparse.RawTextHelpFormatter)
    parser_analysing.add_argument('-a', '--algorithm', type=str, help='Алгоритм, используемый для анализа.\
                                 Поддерживаемые алгоритмы стегоанализа: visual.')
    parser_analysing.add_argument('-p', '--params', type=str, help='Параметры алгоритма, используемого для анализа.')
    parser_analysing.add_argument('-s', '--stego', type=str, help='Файл, который будет подвергнут стеганографическому анализу (предполагаемая стеганограмма).\
                                 Поддерживаемые типы файлов: bmp.')
    parser_analysing.add_argument('-r', '--result', type=str, help='Файл с результатами анализа. Тип файла, зависит от используемого алгоритма.')
    parser_analysing.set_defaults(func=analysing)

    args = parser.parse_args()
    
    app_config = AppConfig(config_file=args.config)
    app_config.apply_config()

    args.func(args, app_config)

    exit()

