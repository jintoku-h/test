from time import sleep
import signal
import sys

from Dobot.dobot_control import DobotController
import autokobopy as ak

digital_input = True
pins = [1, 10]
Tune_time = 12 #(秒単位)

com_port = ak.parse_str_csv('./config/connect_set.csv')

try:
    dobot = dobot = DobotController(com_port['dobot_ip'], pins)
    zstage = ak.ZStageSuruga(port = com_port['zstage_port'])
    
    dobot.initialize_mg400(pin_hand=1)
    zstage.initialize_stage()

    zstage.go_abs(-24500)

    #ホモジナイザー操作、pin番号を引数で入れる
    dobot.io_on_off(io_pin=9, time=Tune_time)
    
    zstage.go_home()

    print("チューニング終了")
except Exception as e:
    print(f'error:{e}')

    
