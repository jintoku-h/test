"""
粉体ディスペンサーの動作確認のためのコード
"""
from time import sleep
import numpy as np
import pandas as pd
import signal
import sys

import autokobopy as ak
from autokobopy import ipconnect as ip

# CSVから通信ポート設定を読み込み
com_port = ak.parse_str_csv('./config/connect_set.csv')

# ディスペンサーのインスタンス作成
ser = [
    ak.PwDispenserSDB1(com_port['powder_disp0_port'], firmware=3.7),
    ak.PwDispenserSDB1(com_port['powder_disp1_port'], firmware=3.7),
    ak.PwDispenserSDB1(com_port['powder_disp2_port'], firmware=3.7)
]

#print(ak.status_query)

# 複数の位置と対応する粉体カウント数（例として3つ）
powder_dispenser_positions = [2] #0, 1, 2ディスペンサーの番号
powder_counts = [5] # 吐出回数

# 秤量処理ループ
for position, count in zip(powder_dispenser_positions, powder_counts):
    sleep(0.5)
    print(f"powder{count}")
    ser[position].set_vibration_time(0) #0~5 sec
    ser[position].set_motor_speed(2) # 0: fast, 1: normal, 2: slow
    #ser[position].set_motor_stop_time(5)
    ser[position].status_query()
    ser[position].shot_several(int(count))
    print("finish")

# for position, count in zip(powder_dispenser_positions, powder_counts):
#     sleep(1)
#     print(f"powder{count}")
#     ser[position].shot_several(int(count))
#     print("finish")

# for position, count in zip(powder_dispenser_positions, powder_counts):
#     sleep(1)
#     print(f"powder{count}")
#     ser[position].shot_several(int(count))
#     print("finish")