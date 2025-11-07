from time import sleep
import numpy as np
import pandas as pd
import signal
import sys

from Dobot.dobot_control import DobotController
import autokobopy as ak
from autokobopy import ipconnect as ip

digital_input = True
pins = [1,9,10]
input_pin = 2


def signal_handler(sig, frame):
    print('\nCtrl+C detected. Stopping execution...')
    sys.exit(0)

def main_process():

    #バイアルホルダーのパレット作成
    global neutral_position
    parameters = ak.parse_float_csv('./config/robot_param_skip_cap.csv') #ここにティーチング座標を入力したcsvファイルの名前
    com_port = ak.parse_str_csv('./config/connect_set.csv')
    table = np.empty((5,4))
    cap_opener_position = np.empty(4)
    liq_dispenser_position = np.empty((3,4))
    powder_dispenser_position = np.empty((3,4))
    z_stage_position = np.empty(4)
    neutral_position = np.empty(4)
    

    #1軸目の個数
    table[0][0] = parameters['holder_holes_row']
    #2軸目の個数
    table[1][0] =  parameters['holder_holes_line']
    
    #パレットの原点
    table[2][0] = parameters['holder_origin_position_x']
    table[2][1] = parameters['holder_origin_position_y']
    table[2][2] = parameters['holder_origin_position_z']
    table[2][3] = parameters['holder_origin_position_r']

    #1軸目の基準点
    table[3][0]  = parameters['holder_corner_top_left_x']
    table[3][1] = parameters['holder_corner_top_left_y']
    table[3][2]  = parameters['holder_corner_top_left_z']
    table[3][3] = parameters['holder_corner_top_left_r']

    #2軸目の基準点
    table[4][0] = parameters['holder_corner_bottom_right_x']
    table[4][1] = parameters['holder_corner_bottom_right_y']
    table[4][2] = parameters['holder_corner_bottom_right_z'] 
    table[4][3] = parameters['holder_corner_bottom_right_r']
    print(table)

    cap_opener_position[0] = parameters['cap_opener_position_x']
    cap_opener_position[1] = parameters['cap_opener_position_y']
    cap_opener_position[2] = parameters['cap_opener_position_z']
    cap_opener_position[3] = parameters['cap_opener_position_r']
    
    #液体ディスペンサーの位置はスライダ―固定でMG400側の位置を調整すると良い
    liq_dispenser_position[0][0] = parameters['liq_dispenser_position_1_x']
    liq_dispenser_position[0][1] = parameters['liq_dispenser_position_1_y']
    liq_dispenser_position[0][2] = parameters['liq_dispenser_position_1_z']
    liq_dispenser_position[0][3] = parameters['liq_dispenser_position_1_r']

    liq_dispenser_position[1][0] = parameters['liq_dispenser_position_2_x']
    liq_dispenser_position[1][1] = parameters['liq_dispenser_position_2_y']
    liq_dispenser_position[1][2] = parameters['liq_dispenser_position_2_z']
    liq_dispenser_position[1][3] = parameters['liq_dispenser_position_2_r']

    liq_dispenser_position[2][0] = parameters['liq_dispenser_position_3_x']
    liq_dispenser_position[2][1] = parameters['liq_dispenser_position_3_y']
    liq_dispenser_position[2][2] = parameters['liq_dispenser_position_3_z']
    liq_dispenser_position[2][3] = parameters['liq_dispenser_position_3_r']

    #粉体ディスペンサーの位置はMG400側は固定でスライダーの位置を調整すると良い
    powder_dispenser_position[0][0] = parameters['powder_dispenser_position_1_x']
    powder_dispenser_position[0][1] = parameters['powder_dispenser_position_1_y']
    powder_dispenser_position[0][2] = parameters['powder_dispenser_position_1_z']
    powder_dispenser_position[0][3] = parameters['powder_dispenser_position_1_r']

    powder_dispenser_position[1][0] = parameters['powder_dispenser_position_2_x']
    powder_dispenser_position[1][1] = parameters['powder_dispenser_position_2_y']
    powder_dispenser_position[1][2] = parameters['powder_dispenser_position_2_z']
    powder_dispenser_position[1][3] = parameters['powder_dispenser_position_2_r']

    powder_dispenser_position[2][0] = parameters['powder_dispenser_position_3_x']
    powder_dispenser_position[2][1] = parameters['powder_dispenser_position_3_y']
    powder_dispenser_position[2][2] = parameters['powder_dispenser_position_3_z']
    powder_dispenser_position[2][3] = parameters['powder_dispenser_position_3_r']


    #z軸ステージの位置
    z_stage_position[0] = parameters['z_stage_position_x']
    z_stage_position[1] = parameters['z_stage_position_y']
    z_stage_position[2] = parameters['z_stage_position_z']
    z_stage_position[3] = parameters['z_stage_position_r']

    #各操作の前後に必ずとらせる姿勢(MG400のみ)
    neutral_position[0] = parameters['neutral_position_x']
    neutral_position[1] = parameters['neutral_position_y']
    neutral_position[2] = parameters['neutral_position_z']
    neutral_position[3] = parameters['neutral_position_r']

    try:
        #MG400との接続
        dobot = DobotController(com_port['dobot_ip'], pins)
        
        #Z軸ステージとの接続
        zstage = ak.ZStageSuruga(port = com_port['zstage_port'])

        dobot.initialize_mg400(pin_hand=1)
        dobot.run_movJ(neutral_position)
        sleep(1)

        zstage.go_abs(-1000)

        zstage.initialize_stage()

        """
        #キャップオープナーのコントローラーへ接続
        cap_opener_host = com_port['edge_ip']
        port = 12345
        sock = ip.connect_to_server(cap_opener_host,port)

        #キャップオープナーの初期化
        ip.send_command(sock,"INIT")
        dobot.release()
        """

        #ディスペンサー類との接続
        sender = ak.LqDispenserXLP6000(com_port['liquid_disp_port']) #液体ディスペンサー

        ser = [ak.PwDispenserSDB1(com_port['powder_disp0_port'], firmware=3.7), ak.PwDispenserSDB1(com_port['powder_disp1_port'], firmware=3.7), 
               ak.PwDispenserSDB1(com_port['powder_disp2_port'], firmware=3.7)] #粉体ディスペンサー
        #左から順番にpowder0,powder1,powder2と対応する
        
        pallet = ak.create_4D_pallet(table)
        print(pallet)
        index = np.empty(2)
        dispens_index = 0
        powder_index = 0

        #"""
        # ディスペンサー動作用ファイルの読み込み
        df = pd.read_csv('config/experimental_param.csv')

        milliliters = np.empty((3,18))
        milliliters[0] = df["liquid0_milliliters"]
        milliliters[1] = df["liquid1_milliliters"]
        milliliters[2] = df["liquid2_milliliters"]

        print(milliliters)
        max_milliliters = [10.0, 25.0, 25.0]
        
        
        for i in range(3):
            ser[i].set_motor_stop_time(100)
        

        powder_count = np.empty((3,18))
        powder_count[0] = df["powder0_count"]
        powder_count[1] = df["powder1_count"]
        powder_count[2] = df["powder2_count"]

        print(powder_count)


        homogenizer_time = df["homogenizer_time"]
        
        zstage.go_home()

        #"""
        print("let")
    
        for i in range(int(table[0][0])):
            for j in range(int(table[1][0])):
                #"""
                index[0] = i
                index[1] = j
                #パレットからピック

                dobot.pick_holder(ak.pallet_position(pallet,index))

                """
                if not dobot.grasp_or_not(input_pin):
                    dobot.release()
                    ip.send_command(sock,"INIT")
                    continue

                dobot.place_opener(cap_opener_position,sock)
                ip.send_command(sock,"GRAB_CAP")
                ip.send_command(sock,"OPEN_CAP")


                dobot.pick_opener(cap_opener_position - [0,0,20,0],sock)
                if not dobot.grasp_or_not(input_pin):
                    dobot.release()
                    ip.send_command(sock,"INIT")
                    continue
                """


                dobot.run_movJ(neutral_position)
                sleep(1)


                
                
                #粉体秤量動作
                for k in range(3):
                    if not (int(powder_count[k][powder_index])) == 0:
                        if not dobot.grasp_or_not(input_pin):
                            dobot.release()
                            # ip.send_command(sock,"INIT")
                            break
                        dobot.leave_from_dispenser(powder_dispenser_position[k])
                        dobot.move_to_dispenser(powder_dispenser_position[k])
                        sleep(1)
                        print("powder{}".format(powder_count[k][powder_index]))
                        ser[k].shot_several(int(powder_count[k][powder_index]))
                        dobot.leave_from_dispenser(powder_dispenser_position[k])
                        print(k)
                

                #液体分注動作
                for m in range(3):
                    if not (milliliters[m][dispens_index]) == 0:
                        if not dobot.grasp_or_not(input_pin):
                            dobot.release()
                            # ip.send_command(sock,"INIT")
                            break
                        dobot.leave_from_dispenser(liq_dispenser_position[m])
                        dobot.move_to_dispenser(liq_dispenser_position[m])
                        sleep(1)
                        sender.aspirate_milliliters(m, milliliters[m][dispens_index], max_milliliters[m])
                        sender.dispense_milliliters(m, milliliters[m][dispens_index], max_milliliters[m])
                        dobot.leave_from_dispenser(liq_dispenser_position[m])
                        print(m)

                dobot.run_movJ(neutral_position)
                sleep(1)

                if not dobot.grasp_or_not(input_pin):
                    dobot.release()
                    # ip.send_command(sock,"INIT")
                    continue

                #Z軸ステージ運搬&ホモジナイザー操作
                dobot.place_zstage(z_stage_position, zstage, -7800)

                zstage.go_abs(-24500)

                #ホモジナイザー操作、pin番号を引数で入れる
                dobot.io_on_off(io_pin=9, time=homogenizer_time[dispens_index])
                
                zstage.go_abs(-6800)

                dobot.pick_zstage(z_stage_position, zstage)

                zstage.go_abs(-200)

                if not dobot.grasp_or_not(input_pin):
                    dobot.release()
                    # ip.send_command(sock,"INIT")
                    print("ホモジナイザーのところで取り損ねました。ごめんなさい")
                    raise Exception
                    
                
                """
                #キャップを締めてホルダーへ戻す
                dobot.place_opener(cap_opener_position - [0,0,20,0],sock)
                ip.send_command(sock,"CLOSE_CAP")

                dobot.pick_opener(cap_opener_position - [0,0,7,0],sock)

                if not dobot.grasp_or_not(input_pin):
                    print("Cap Closing failed")
                    raise
                """

                dobot.place_holder(ak.pallet_position(pallet,index))

                dispens_index = dispens_index + 1
                powder_index = powder_index + 1
                print(powder_index, dispens_index)
                
    except Exception as e:
        print(f"An error occurred during communication: {e}")
        if dobot:
            dobot.stop_robot()
        
        
    finally:
        if dobot:
            dobot.stop_robot()

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    try:
        main_process()
    except SystemExit:
        print("Program terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Cleanup complete. Exiting program.")
