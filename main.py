import logging
import argparse
from argparse import Namespace

# from numpy import set_printoptions, inf

from config import AppCongig
from LSB import LSB_embedding, LSB_extracting
from steganalysis import visual_attack


# set_printoptions(threshold=inf)
# set_printoptions(linewidth=inf)


def embeding(args: Namespace, app_config: AppCongig):
    cover_file_path = app_config.get_covers_file_path(args.cover)
    stego_file_path = app_config.get_stegos_file_path(args.stego)
    message_file_path = app_config.get_messages_file_path(args.message)
    algorithm = args.algorithm

    match algorithm:
        case 'lsb':
            stego_vect = LSB_embedding(
                app_config,
                cover_file_path,
                stego_file_path,
                message_file_path,
                fill_rest=True
            )
    #     case '2':
    #         stego_vect = LSB_PRI_embedding(cover_vect, message_bits, 1)


def extracting(args: Namespace, app_config: AppCongig):
    stego_file_path = app_config.get_stegos_file_path(args.stego)
    algorithm = args.algorithm

    match algorithm:
        case 'lsb':
            message_bits = LSB_extracting(app_config, stego_file_path)


def analysis(args: Namespace, app_config: AppCongig):
    algorithm = args.algorithm
    stego_file_path = app_config.get_stegos_file_path(args.stego)
    result_file_path = app_config.get_analysis_file_path(args.result)

    match algorithm:
        case 'visual':
            visual_attack(app_config, stego_file_path, result_file_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Стеганографическое погружение/извлечение/анализ информации.')
    parser.add_argument('-cfg', '--config', type=str, help='Путь к файлу конфигурации в формате toml.', default=None)
    subparsers = parser.add_subparsers()

    parser_embed = subparsers.add_parser('embed', help='Погружение информации.')
    parser_embed.add_argument('-c', '--cover', type=str, help='Путь к файлу - покрывающему объекту (контейнеру),\
                               в который осуществляется погружение (встраивание) вложения (скрываемой информации).\
                              Поддерживаемые типы файлов: bmp.')
    parser_embed.add_argument('-m', '--message', type=str, help='Путь к файлу вложению (скрываемой информации),\
                              которое погружается (встраивается) в покрывающий объект (контейнер).\
                              Поддерживаются любые типы файлов.')
    parser_embed.add_argument('-a', '--algorithm', type=str, help='Алгоритм, используемый для погружения вложения в покрывающий объект.\
                              Поддерживаемые алгоритмы: lsb.')
    parser_embed.add_argument('-s', '--stego', type=str, help='Путь к файлу - стеганограмме результату погруженния вложения в покрывающий объект.')
    parser_embed.set_defaults(func=embeding)
    
    parser_extract = subparsers.add_parser('extract', help='Извлечение информации.')
    parser_extract.add_argument('-s', '--stego', type=str, help='Стеганограмма из которого осуществляется извлечение вложения (скрытой в ней информации).\
                                Поддерживаемые типы файлов: bmp.')
    parser_extract.add_argument('-a', '--algorithm', type=str, help='Алгоритм, который был использован для погружения.\
                                Поддерживаемые алгоритмы: lsb.')
    parser_extract.set_defaults(func=extracting)

    parser_analysis = subparsers.add_parser('analysis', help='Стеганографический анализ.')
    parser_analysis.add_argument('-s', '--stego', type=str, help='Файл, который будет подвергнут стеганографическому анализу (предполагаемая стеганограмма).\
                                 Поддерживаемые типы файлов: bmp.')
    parser_analysis.add_argument('-a', '--algorithm', type=str, help='Алгоритм, используемый для анализа.\
                                 Поддерживаемые алгоритмы стегоанализа: visual.')
    parser_analysis.add_argument('-r', '--result', type=str, help='Файл с результатами анализа. Тип файла, зависит от используемого алгоритма.')
    parser_analysis.set_defaults(func=analysis)

    args = parser.parse_args()
    
    app_config = AppCongig(config_file=args.config)
    app_config.apply_config()

    args.func(args, app_config)

    exit()

