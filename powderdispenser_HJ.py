import serial
import time

#########################
#pwdispenser.py
#2025/03/13
#
#Yuto Mueller
#########################

########################################
#@class PwDispenserSDB1
#@breif オーバーライドするのクラスをファームウェアのバージョンによって決定する
########################################
def PwDispenserSDB1(port : str, baudrate=19200, timeout=1, firmware:float=3.7):
    if firmware >= 3.8:
        return PwDispenserSDB1_firm40(port=port, baudrate=baudrate, timeout=timeout)
    elif firmware == 3.7:
        return PwDispenserSDB1_firm37(port=port, baudrate=baudrate, timeout=timeout)

########################################
#@class PwDispenserSDB1_Base
#@breif SDB-1制御用の親クラス
########################################

class PwDispenserSDB1_Base:
    #########################################
    #@name __init__
    #@breif インスタンス化の際にUSBシリアル通信を確立する
    #@ret なし
    #########################################
    def __init__(self, port : str, baudrate=19200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        try:
            self._ser = serial.Serial(port, baudrate, bytesize=8, stopbits=1, timeout=timeout)
            print("SDB-1：シリアル通信の確立に成功")
            time.sleep(2)
        except serial.SerialException as e:
            print(f"SDB-1：シリアル通信の確立に失敗: {e}")

    #########################################
    #@name close
    #@breif USBシリアル通信を切断する
    #@ret なし
    #########################################
    def close(self):
        self._ser.close()
        print("SDB-1：シリアル通信を切断しました")

    #########################################
    #@name __del__
    #@breif オブジェクトが削除された際にclose関数を呼び出す
    #@ret なし
    #########################################
    def __del__(self):
        self.close()

    #########################################
    #@name _send_data
    #@breif ディスペンサーがビジー状態から解放されるまで待機してからデータを送信する。
    #@ret response :正常応答
    #########################################
    def _send_data(self, data: bytes):
        """
        ビジー状態でのタイムアウトを防ぐために、ビジーチェックを強化。
        """
        max_retries = 5
        retry_delay = 3  # ビジー状態のときの待機時間
        retries = 0

        while retries < max_retries:
            self._ser.write(bytes(data))
            time.sleep(0.5)
            response = self._wait_reply()
            
            if response:
                # 応答を確認してビジー状態かどうかを判断
                if response[0] == 0x16:  # ビジー応答
                    print("SDB-1：ビジー状態です。再試行中...")
                    time.sleep(retry_delay)
                    retries += 1
                elif response[0] == 0x15:  # NAK応答
                    print("SDB-1：NAKエラーを受信しました。再試行中...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    #print("信号送信成功")
                    return response  # 正常応答を取得
            else:
                print("SDB-1：応答なし。再試行中...")
                time.sleep(retry_delay)
                retries += 1
        
        # 最大リトライ回数に達した場合
        print("SDB-1：最大リトライ回数に達しました。ビジー状態が解除されません。")
        return None

    #########################################
    #@name _send_data
    #@breif ディスペンサーからの応答を待つ。ビジー状態の場合は待機し、リトライする。
    #@ret buffer:受信内容
    #########################################
    def _wait_reply(self):
        start_time = time.time()
        time_out = 5
        buffer = []

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > time_out:
                print("SDB-1：信号を受け取れずにタイムアウトしました")
                return None

            try:
                if self._ser.in_waiting > 0:
                    # データを受信してバッファに追加
                    data = self._ser.read(self._ser.in_waiting)
                    buffer.extend(self._bytes2int_list(data))

                    if buffer and buffer[-1] == 0x5A:
                        #print(f"受信完了: {buffer}")
                        return buffer
                    elif buffer and buffer[-1] == 0xFE:
                        return buffer    

            except serial.SerialTimeoutException:
                print("SDB-1：読み取り中にタイムアウトが発生しました")
                return None

    #########################################
    #@name _bytes2int_list
    #@breif コマンドバイトをint型のリストに変換する
    #@ret data_int_list:コマンドバイトをint型のリストにしたもの
    #########################################
    def _bytes2int_list(self, data : bytes):
        data_hex = data.hex()
        data_int_list = [int(data_hex[i:i+2], 16) for i in range(0, len(data_hex), 2)]
        return data_int_list

    #########################################
    #@name _send_data_query
    #@breif　データを送信する命令を出す
    #@ret なし
    #########################################
    def _send_data_query(self, data : bytes):       
        self._ser.write(data)

    #########################################
    #@name _wait_reply_query
    #@breif　制御対象からの応答を待つ命令を出す
    #@ret なし
    #########################################
    def _wait_reply_query(self, end_sig_list=None, timeout=10):
        start_time = time.time()
        buffer = bytearray()

        while True:
            if time.time() - start_time > timeout:
                print("信号を受け取れずにタイムアウトしました")
                return buffer

            if self._ser.in_waiting > 0:
                data = self._ser.read_all()
                buffer.extend(data)

                # 終了信号を受信したら抜ける
                if end_sig_list and buffer[-len(end_sig_list):] == bytes(end_sig_list):
                    return buffer


    #########################################
    #@name shot_onetime
    #@breif 一度だけ吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_onetime(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    #########################################
    #@name shot_several
    #@breif 複数回吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_several(self, count : int):
        raise NotImplementedError("This method should be overridden by subclasses")

    #########################################
    #@name status_query
    #@breif 現在の各種設定を確認する。
    #@ret なし
    #########################################
    def status_query(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    #########################################
    #@name set_wait_time
    #@breif 吐出間隔を設定する
    #@ret なし
    #########################################
    def set_wait_time(self, waittime : int):
        raise NotImplementedError("This method should be overridden by subclasses")

    #########################################
    #@name set_motor_speed
    #@breif モーターのスピードを設定する
    #@ret なし
    #########################################
    def set_motor_speed(self, level: int):
        raise NotImplementedError("This method should be overridden by subclasses")
   
    #########################################
    #@name set_vibration_time
    #@breif 吐出後のバイブレーターの稼働時間を設定する
    #@ret なし
    #########################################
    def set_vibration_time(self, vibrationtime : int):
        raise NotImplementedError("This method should be overridden by subclasses")
    #########################################
    #@name set_motor_stop_time
    #@breif モーター停止位置を設定する
    #@ret なし
    #########################################
    def set_motor_stop_time(self, time : int):
        raise NotImplementedError("This method should be overridden by subclasses")
    
########################################
#@class PwDispenserSDB1_firm37
#@breif SDB-1(Firmware=3.7)制御用の子クラス
########################################
class PwDispenserSDB1_firm37(PwDispenserSDB1_Base):
    #########################################
    #@name __init__
    #@breif インスタンス化の際にUSBシリアル通信を確立する親クラスのinitを実行する
    #@ret なし
    #########################################
    def __init__(self, port : str, baudrate=19200, timeout=1):
            super().__init__(port, baudrate, timeout=timeout)

    #########################################
    #@name __del__
    #@breif オブジェクトが削除された際にclose関数を呼び出す
    #@ret なし
    #########################################
    def __del__(self):
        super().close()

    #########################################
    #@name shot_onetime
    #@breif 一度だけ吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_onetime(self):
        start_sig = bytes([0xA5, 0x44, 0xF1, 0x36, 0x5A])
        end_sig   = bytes([0xA5, 0x44, 0xF5, 0x3D, 0x5A])

        super()._send_data(start_sig)
        print("送信しました:", start_sig)

        buffer = bytearray()

        while True:
            data = super()._wait_reply_query()
            if not data:
                print("受信なし／タイムアウトで終了")
                break

            buffer.extend(data)
            print(f"受信: {data.hex()} (len={len(data)})")

            while len(buffer) >= 5:
                frame = bytes(buffer[:5])
                buffer = buffer[5:]

                if frame == end_sig:
                    print(">>> 終了信号検出（正しい終了）")
                    print("※ 振動が完了するまで3秒待機します")
                    time.sleep(3)  # ← 通信切断を遅らせる
                    return
                else:
                    print(f"中間信号: {frame.hex()}")




    #########################################
    #@name shot_several
    #@breif 複数回吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_several(self, count : int):
        if count == 0:
            print("SDB-1：投入回数の指定が0回です。投入をスキップします")
        else:
            for i in range(count):
                print("SDB-1：投入開始{}回目".format(i+1))
                self.shot_onetime()
                print("SDB-1：投入終了{}回目".format(i+1))

    #########################################
    #@name status_query
    #@breif 現在の各種設定を確認する。
    #@ret なし
    #########################################
    def status_query(self):
        motorspeed_dict = {0:'Fast', 63:'Normal', 127:'Slow'}
        sig_list =         [0xA5, 0x51, 0x00, 0x51, 0x5A]
        status_query_sig = [0xA5, 0x51, 0x00, 0x51, 0x5A]
        super()._send_data_query(status_query_sig)
        time.sleep(0.5)
        read_list = super()._wait_reply_query()
        if all(i in read_list for i in sig_list):
            print("Wait Timeの設定:{}秒 \nMotor Speedの設定:{} \nVibration Timeの設定:{}秒 \nモーター停止位置の設定:{}ms"
                  .format(read_list[6]-48, motorspeed_dict[read_list[7]], read_list[8]-129, read_list[9]-83))
            time.sleep(1)
        else:
            print(read_list)
            print("SDB-1：各設定の確認、失敗")

    #########################################
    #@name set_wait_time
    #@breif 吐出間隔を設定する
    #@ret なし
    #########################################
    def set_wait_time(self, waittime : int):

        waittime_sig = [0xA5, 0x23, 48 + waittime, 83 + waittime, 0x5A]
        read_data = super()._send_data(waittime_sig)
        if read_data == waittime_sig:
            print("SDB-1：Wait Timeの設定成功{}秒".format(waittime))
        else:
            print("SDB-1：Wait Timeの設定失敗")

    #########################################
    #@name set_motor_speed
    #@breif モーターのスピードを設定する
    #@ret なし
    #########################################
    def set_motor_speed(self, level: int):
        motorspeed_dict = {0:(0, 'Fast'), 1:(63, 'Normal'), 2:(127, 'Slow')}
        motorspeed_sig = [0xA5, 0x64, motorspeed_dict[level][0], 100 + motorspeed_dict[level][0], 0x5A]
        read_data = super()._send_data(motorspeed_sig)
        if read_data == motorspeed_sig:
            print("SDB-1：Motor Speedの設定成功{}".format(motorspeed_dict[level][1]))
        else:
            print("SDB-1：Motor Speedの設定失敗")
   
    #########################################
    #@name set_vibration_time
    #@breif 吐出後のバイブレーターの稼働時間を設定する
    #@ret なし
    #########################################
    def set_vibration_time(self, vibrationtime : int):
        """
        振動時間の設定(0~5秒：1秒単位)
        """
        vibrationtime_sig = [0xA5, 0x65, 129 + vibrationtime, 230 + vibrationtime, 0x5A]
        read_data = super()._send_data(vibrationtime_sig)
        if read_data == vibrationtime_sig:
            print("SDB-1：Vibration Timeの設定成功{}秒".format(vibrationtime))
        else:
            print("SDB-1：Vibration Timeの設定失敗")

    #########################################
    #@name set_motor_stop_time
    #@breif モーター停止位置を設定する
    #@ret なし
    #########################################
    def set_motor_stop_time(self, time : int):
        if 185 + time >= 256:
            stop_time_sig = [0xA5, 0x66, 83 + time, 185 + time - 256, 0x5A]
        else:
            stop_time_sig = [0xA5, 0x66, 83 + time, 185 + time, 0x5A]
        read_data = super()._send_data(stop_time_sig)
        if read_data == stop_time_sig:
            print("SDB-1：モーター停止位置の設定成功{}ms".format(time))
        else:
            print("SDB-1：モーター停止位置の設定失敗")

########################################
#@class PwDispenserSDB1_firm40
#@breif SDB-1(Firmware=4 > )制御用の子クラス
########################################
class PwDispenserSDB1_firm40(PwDispenserSDB1_Base):
    #########################################
    #@name __init__
    #@breif インスタンス化の際にUSBシリアル通信を確立する親クラスのinitを実行する
    #@ret なし
    #########################################
    def __init__(self, port : str, baudrate=19200, timeout=1):
            super().__init__(port, baudrate, timeout=timeout)

    #########################################
    #@name __del__
    #@breif オブジェクトが削除された際にclose関数を呼び出す
    #@ret なし
    #########################################
    def __del__(self):
        super().close()

    #########################################
    #@name shot_onetime
    #@breif 一度だけ吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_onetime(self):
        start_sig =    [0xEF, 0x46, 0x01, 0x47, 0xFE]
        end_sig_list = [0xEF, 0x46, 0x01, 0x47, 0xFE]
        while True:
            try:
                # データを受け取るまで待機
                received_list = super()._send_data(start_sig)
                time.sleep(1)
                # 終了信号を受け取るまで待機
                if all(i in received_list for i in end_sig_list):
                    print("Shot Complete")
                    break
            except serial.SerialTimeoutException:
                print("SDB-1：読み取り中にタイムアウトが発生しました")

    #########################################
    #@name shot_several
    #@breif 複数回吐出を行う指示を出す
    #@ret なし
    #########################################
    def shot_several(self, count : int):
        if count == 0:
            print("SDB-1：投入回数の指定が0回です。投入をスキップします")
        else:
            for i in range(count):
                print("SDB-1：投入開始{}回目".format(i+1))
                self.shot_onetime()
                print("SDB-1：投入終了{}回目".format(i+1))

    #########################################
    #@name status_query
    #@breif 現在の各種設定を確認する。
    #@ret なし
    #########################################
    def status_query(self):
        motorspeed_dict = {0:'Fast', 63:'Normal', 127:'Slow'}
        sig_list =         [0xEF, 0x51, 0x00, 0x51, 0xFE]
        status_query_sig = [0xEF, 0x51, 0x00, 0x51, 0xFE]
        super()._send_data_query(status_query_sig)
        time.sleep(0.5)
        read_list = super()._wait_reply_query()
        if all(i in read_list for i in sig_list):
            print("Wait Timeの設定:{}秒 \nMotor Speedの設定:{} \nVibration Timeの設定:{}秒 \nモーター停止位置の設定:{}ms"
                  .format(read_list[6]-48, motorspeed_dict[read_list[7]], read_list[8]-129, read_list[9]-83))
            time.sleep(1)
        else:
            print(read_list)
            print("SDB-1：各設定の確認、失敗")

    #########################################
    #@name set_wait_time
    #@breif 吐出間隔を設定する
    #@ret なし
    #########################################
    def set_wait_time(self, waittime : int):

        waittime_sig = [0xEF, 0x23, 48 + waittime, 83 + waittime, 0xFE]
        read_data = super()._send_data(waittime_sig)
        all(i in read_data for i in waittime_sig)
        if read_data == waittime_sig:
            print("SDB-1：Wait Timeの設定成功{}秒".format(waittime))
        else:
            print("SDB-1：Wait Timeの設定失敗")

    #########################################
    #@name set_motor_speed
    #@breif モーターのスピードを設定する
    #@ret なし
    #########################################
    def set_motor_speed(self, level: int):
        motorspeed_dict = {0:(0, 'Fast'), 1:(63, 'Normal'), 2:(127, 'Slow')}
        motorspeed_sig = [0xEF, 0x64, motorspeed_dict[level][0], 100 + motorspeed_dict[level][0], 0xFE]
        read_data = super()._send_data(motorspeed_sig)
        if read_data == motorspeed_sig:
            print("SDB-1：Motor Speedの設定成功{}".format(motorspeed_dict[level][1]))
        else:
            print("SDB-1：Motor Speedの設定失敗")
   
    #########################################
    #@name set_vibration_time
    #@breif 吐出後のバイブレーターの稼働時間を設定する
    #@ret なし
    #########################################
    def set_vibration_time(self, vibrationtime : int):
        """
        振動時間の設定(0~5秒：1秒単位)
        """
        vibrationtime_sig = [0xEF, 0x65, 129 + vibrationtime, 230 + vibrationtime, 0xFE]
        read_data = super()._send_data(vibrationtime_sig)
        if read_data == vibrationtime_sig:
            print("SDB-1：Vibration Timeの設定成功{}秒".format(vibrationtime))
        else:
            print("SDB-1：Vibration Timeの設定失敗")

    #########################################
    #@name set_motor_stop_time
    #@breif モーター停止位置を設定する
    #@ret なし
    #########################################
    def set_motor_stop_time(self, time : int):
        if 185 + time >= 256:
            stop_time_sig = [0xEF, 0x66, 83 + time, 185 + time - 256, 0xFE]
        else:
            stop_time_sig = [0xEF, 0x66, 83 + time, 185 + time, 0xFE]
        read_data = super()._send_data(stop_time_sig)
        if read_data == stop_time_sig:
            print("SDB-1：モーター停止位置の設定成功{}ms".format(time))
        else:
            print("SDB-1：モーター停止位置の設定失敗")
