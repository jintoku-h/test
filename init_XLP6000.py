from time import sleep
import numpy as np
import pandas as pd
import signal
import sys

import autokobopy as ak
from autokobopy import ipconnect as ip
from lqdispenser_HJ import LqDispenserXLP6000

def initialize_process():
    com_port = ak.parse_str_csv('./config/connect_set.csv')
    sender = ak.LqDispenserXLP6000(com_port['liquid_disp_port'])

    sender.initialize_pump_async(0)
    sender.initialize_pump_async(1)
    sender.initialize_pump_async(2)

    print("Initializing done!")

def main_process():
    program_string  = "YS14IA100OA0I" # A以下の数字がステップ数 Max6000 "YS14IA1000OA0I"
    com_port = ak.parse_str_csv('./config/connect_set.csv') 
    sender = LqDispenserXLP6000(com_port['liquid_disp_port'])

    for _ in range(1):
        for ch in [0,1,2]:
            sender.initialize_pump(ch, program_string)

    print("初期位置への移動が完了しました")

if __name__ == "__main__":
    initialize_process()
    main_process()