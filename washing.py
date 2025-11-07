from time import sleep
import numpy as np
import pandas as pd
import signal
import sys

import autokobopy as ak
from autokobopy import ipconnect as ip
from lqdispenser_HJ import LqDispenserXLP6000

com_port = ak.parse_str_csv('./config/connect_set.csv')
sender = LqDispenserXLP6000(com_port['liquid_disp_port'])

def substitute_process():
    program_string  = "YS14IA1000OA0I" # A以下の数字がステップ数 Max6000 "YS14IA1000OA0I"
    # 置換時 10mL: 1000 x 2 set     25 mL: 1000 x 2 set

    # 指定のポンプを指定回数動作して洗浄
    for _ in range(2):
        for ch in [0,1]:
            sender.initialize_pump(ch, program_string)

    print("溶液置換が完了しました")


def washing_process():
    program_string  = "YS14IA2000OA0I" # A以下の数字がステップ数 Max6000 "YS14IA1000OA0I"
    # 洗浄時 10mL: 2000 x 5 set     25 mL: 2000 x 3 set

    for _ in range(3):
        for ch in [0,1]:
            sender.initialize_pump(ch, program_string)

    print("洗浄が完了しました")

if __name__ == "__main__":
    #substitute_process()
    washing_process()