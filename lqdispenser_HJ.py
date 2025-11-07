import serial
import time

#########################
#lqdispenser.py
#2025/01/15
#
#Yuto Mueller
#########################

########################################
#@class LqDispenserXLP6000
#@breif XLP6000制御用のクラス
########################################
class LqDispenserXLP6000:
    #########################################
    #@name __init__
    #@breif インスタンス化の際にUSBシリアル通信を確立する
    #@ret なし
    #########################################
    def __init__(self, port, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.sequence_number = 0
        self.previous_command = None
        
        try:
            self._ser = serial.Serial(port, baudrate, bytesize=8, stopbits=1, timeout=1)
            print("XLP6000に接続完了")
            time.sleep(2)
        except serial.SerialException as e:
            print(f"XLP6000に接続失敗: {e}")
            
    #########################################
    #@name close
    #@breif USBシリアル通信を切断する
    #@ret なし
    #########################################
    def close(self):
        self._stop_pump(0)
        self._stop_pump(1)
        self._stop_pump(2)
        self._ser._close()
        print("XLP6000の通信を切断")
    
    #########################################
    #@name __del__
    #@breif オブジェクトが削除された際にclose関数を呼び出す
    #@ret なし
    #########################################
    def __del__(self):
        self.close()

    #########################################
    #@name _calculate_checksum
    #@breif 
    #@ret 
    #########################################
    def _calculate_checksum(self, data):
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum
    
    #########################################
    #@name _construct_command
    #@breif 制御用のコマンドを構築する
    #@ret frame:送信用のコマンド
    #########################################
    def _construct_command(self, pump_address, command, repeat_flag=False):
        STX = 0x02

        # ポンプアドレスの指定
        pump_address = 0x31 + pump_address

        # シーケンス番号とリピートフラグの組み合わせ
        sequence_number = self.sequence_number & 0x07  #シーケンス番号
        repeat_bit = 0x08 if repeat_flag else 0x00  #リピートフラグ
        sequence_byte = 0x30 | repeat_bit | sequence_number  #0x30は固定ビット "0011"

        data_block = b'' if command == 'Q' else command.encode('ascii')

        # ETX
        ETX = 0x03

        # フレームの構築
        frame = bytearray()
        frame.append(STX)
        frame.append(pump_address)
        frame.append(sequence_byte)
        frame.extend(data_block)
        frame.append(ETX)

        # チェックサムの計算
        checksum = self._calculate_checksum(frame)
        frame.append(checksum)

        return frame

    #########################################
    #@name _send_serial_command_sync
    #@breif 制御装置にコマンドを送信する
    #@ret response:サーバーから読み取った文字列
    #########################################
    def _send_serial_command_sync(self, pump_address, command, repeat_flag=False):
        if self._wait_until_ready(pump_address):
            try:
                # コマンドを構築
                complete_command = f"{command}R"
                frame = self._construct_command(pump_address, complete_command, repeat_flag)

                # コマンドを送信（バイト列として）
                self._ser.write(frame)

                # 応答を読み取る（必要に応じて）
                response = self._ser.read(100)

                # コマンドが成功した場合、シーケンス番号を更新
                if not repeat_flag:
                    self.sequence_number = (self.sequence_number + 1) % 8

                # 前回のコマンドを更新
                self.previous_command = command

                # 応答が空でないか確認
                if self._wait_until_ready(pump_address):
                    return response

            except Exception as e:
                print(f"エラー: {e}")

    #########################################
    #@name _send_serial_command_async
    #@breif 制御装置にコマンドを送信する。ポンプを並列で操作できる
    #@ret response:サーバーから読み取った文字列
    ######################################### 
    def _send_serial_command_async(self, pump_address, command, repeat_flag=False):
        # コマンドを構築
        complete_command = f"{command}R"
        frame = self._construct_command(pump_address, complete_command, repeat_flag)
        # コマンドを送信（バイト列として）
        self._ser.write(frame)
        # 応答を読み取る（必要に応じて）
        response = self._ser.read(100)  # 読み取るバイト数を指定
        return response

    #########################################
    #@name _wait_until_ready
    #@breif ポンプのステータスを確認し、準備ができるまで待機する
    #@ret bool
    #########################################
    def _wait_until_ready(self, pump_address, check_interval=1):
        while True:
            is_busy = self._check_pump_busy(pump_address)
            if is_busy is None:
                print("XLP6000：ポンプ{}のステータスが確認できませんでした".format(pump_address))
                return False
            if not is_busy:
                return True
            print("XLP6000：ポンプ{}が稼働中です。少しお待ちください。".format(pump_address))
            time.sleep(check_interval)

    #########################################
    #@name _check_pump_busy
    #@breif ポンプがビジー状態でないかを確認する
    #@ret is_busy:動作中かどうかを表す
    #########################################
    def _check_pump_busy(self, pump_address:int):
        response = self._send_serial_command_async(pump_address, command="Q")
        if response:
            status_byte = response[3]
            is_busy = (status_byte & 0x20) == 0
            return is_busy
        else:
            return None

    #########################################
    #@name check_pump_status
    #@breif ポンプの状態を確認する
    #@ret is_busy:動作中かどうかを表す
    #@ret error_code:エラーコード
    #########################################
    def check_pump_status(self, pump_address:int):
        error_list = ["０：正常です。エラーはありません", "１：初期化に失敗しています。再度初期化してください。", "２：無効なコマンドです。", "３：無効なパラメータが使用されています。", 
                      "", "", "６：EEPROMのエラーです。メーカー技術サポートに連絡してください。", "７：デバイスが初期化されていません。初期化を行ってください", "８：内部エラーです。メーカー技術サポートに連絡してください。",
                      "９：ブランジャーに過負荷がかかっています。一度初期化を行ってください。", "10:バルブに過負荷がかかっています。一度初期化を行ってください。",
                      "１１：バルブがバイパス状態です。", "12:内部エラーです。メーカー技術サポートに連絡してください。", "13:ブランジャーの移動コマンドが停止されています。一度初期化を行ってください。",
                      "14:A/Dコンバータのエラーです。メーカー技術サポートに連絡してください。", "15:コマンドを実行中です。現在命令を受け付けることはできません。"]
        response = self._send_serial_command_sync(pump_address, command="Q")
        if response:
            status_byte = response[3]
            is_busy = (status_byte & 0x20) == 0
            error_code = status_byte & 0x0F
            if is_busy is False:
                print("XLP6000：ポンプ{}の準備完了".format(pump_address))
            print("XLP6000：ポンプ{}のエラーコード {}".format(pump_address, error_list[error_code]))
            return is_busy, error_code
        else:
            print("XLP6000：ポンプ{}のステータスが確認できませんでした".format(pump_address))
            return None, None

    #########################################
    #@name _stop_pump
    #@breif ポンプを停止させる
    #@ret なし
    #########################################
    def _stop_pump(self, pump_address):
        command = "T"
        self._send_serial_command_async(pump_address, command)


    #########################################
    #@name initialize_pump
    #@breif ポンプを初期化させる
    #@ret なし
    #########################################
    def initialize_pump(self, pump_address:int, program_string):
        
        program_string = program_string #"YS14IA1000OA0I"
        
        self._load_program_string(pump_address, 1, program_string)
        self._execute_program_string(pump_address, 1)

    #########################################
    #@name initialize_pump_async
    #@breif ポンプを初期化させる。初期化中にほかのポンプを制御も可能
    #@ret なし
    #########################################
    def initialize_pump_async(self, pump_address:int, program_string):
        
        program_string = program_string #"YS14IA1000OA0I"
        
        self._load_program_string_async(pump_address, 1, program_string)
        self._execute_program_string_async(pump_address, 1)

    #########################################
    #@name aspirate_milliliters
    #@breif 液量を指定して吸い上げる(通常モード)
    #@ret なし
    #########################################
    def aspirate_milliliters(self, pump_address:int, milliliters:float, max:float):
        if milliliters == 0:
            print("XLP6000：ポンプ{}の指定が0mLです。スキップします".format(pump_address))
        else:
            self._move_valve_to_input(pump_address)
            steps = milliliters / max * 6000
            rounded_steps = round(steps)
            response = self._aspirate_relative_step(pump_address, rounded_steps)
            if response:
                print("XLP6000：ポンプ{}で液体{}mL吸い上げ".format(pump_address,milliliters))

    
    #########################################
    #@name dispense_milliliters
    #@breif 液量を指定して吐き出し(通常モード)
    #@ret なし
    #########################################
    def dispense_milliliters(self, pump_address:int, milliliters:float, max:float):
        if milliliters == 0:
            return
        else:        
            self._move_valve_to_output(pump_address)
            steps = milliliters / max * 6000
            rounded_steps = round(steps)
            response = self._dispense_relative_step(pump_address, rounded_steps)
            if response:
                print("XLP6000：ポンプ{}で液体{}mL投入".format(pump_address,milliliters))

    
    #########################################
    #@name aspirate_milliliters_microstep
    #@breif 液量を指定して吸い上げ(ファインポジショニングモード、マイクロステップモード)
    #@ret なし
    #########################################
    def aspirate_milliliters_microstep(self, pump_address:int, milliliters:float, max:float):
        if milliliters == 0:
            return
        else:
            self._move_valve_to_input(pump_address)
            steps = milliliters / max * 48000
            rounded_steps = round(steps)
            response = self._aspirate_relative_step(pump_address, rounded_steps)
            if response:
                print("XLP6000：ポンプ{}で液体{}mL吸い上げ(マイクロステップ)".format(pump_address,milliliters))

    
    #########################################
    #@name dispense_milliliters_microstep
    #@breif 液量を指定して吐き出し(ファインポジショニングモード、マイクロステップモード)
    #@ret なし
    #########################################
    def dispense_milliliters_microstep(self, pump_address:int, milliliters:float, max:float):
        if milliliters == 0:
            return
        else:        
            self._move_valve_to_output(pump_address)
            steps = milliliters / max * 48000
            rounded_steps = round(steps)
            response = self._dispense_relative_step(pump_address, rounded_steps)
            if response:
                print("XLP6000：ポンプ{}で液体{}mL投入(マイクロステップ)".format(pump_address,milliliters)) 

    #########################################
    #@name set_pump_mode
    #@breif ポンプのモードを設定する
    #@ret なし
    #########################################
    def set_pump_mode(self, pump_address:int, mode:int):
        """
        マイクロステップモードを設定する関数。
        :param mode: 0（通常モード）、1（ファインポジショニングモード）、2（マイクロステップモード）
        """
        pumpmode_dict = {0:'通常モード', 1:'ファインポジショニングモード', 2:'マイクロステップモード'}
        if mode not in [0, 1, 2]:
            raise ValueError("modeの指定が間違っています。必ず0,1,2のどちらかを選択してください。")
        command = f"N{mode}"
        self._send_serial_command_sync(pump_address, command)
        print("XLP6000：ポンプ{}を{}に変更しました".format(pump_address, pumpmode_dict[mode]))

    def _move_absolute_step(self, pump_address:int, position:int, not_busy=False):
        """
        プランジャーを絶対位置に移動する関数。
        :param position: 移動先の絶対位置
        :param not_busy: プランジャーがビジー状態でないことを示すフラグ
        """
        if not (0 <= position <= 48000):
            raise ValueError("XLP6000：ポンプ{}のpositionの指定が間違っています。通常モードであれば0~6000,マイクロステップモードであれば0~48000で指定してください。".format(pump_address))
        command = f"a{position}" if not_busy else f"A{position}"
        response = self._send_serial_command_sync(pump_address, command)
        if response:
            print("XLP6000：ポンプ{}を{}まで移動しました".format(pump_address, position))

    def _aspirate_relative_step(self, pump_address:int, steps:int, not_busy=False):
        """
        プランジャーを相対位置に移動する関数。
        :param not_busy: プランジャーがビジー状態でないことを示すフラグ
        """
        if not (0 <= steps <= 48000):
            raise ValueError("XLP6000：ポンプ{}のstep数の指定が間違っています。通常モードであれば0~6000,マイクロステップモードであれば0~48000で指定してください。".format(pump_address))
        command = f"p{steps}" if not_busy else f"P{steps}"
        response = self._send_serial_command_sync(pump_address, command)
        return response

    def _dispense_relative_step(self, pump_address:int, steps:int, not_busy=False):
        """
        プランジャーを相対位置に移動する関数。
        :param not_busy: プランジャーがビジー状態でないことを示すフラグ
        """
        if not (0 <= steps <= 48000):
            raise ValueError("XLP6000：ポンプ{}のstep数の指定が間違っています。通常モードであれば0~6000,マイクロステップモードであれば0~48000で指定してください。".format(pump_address))
        command = f"d{steps}" if not_busy else f"D{steps}"
        response = self._send_serial_command_sync(pump_address, command)
        return response

    # 不揮発性メモリに関するコマンド
    def _load_program_string(self, pump_address:int, program_number:int, program_string:str):
        """
        プログラム文字列を不揮発性メモリにロードする関数。
        :param program_number: プログラム番号（0-14）
        :param program_string: プログラム文字列
        """
        if not (0 <= program_number <= 14):
            raise ValueError("XLP6000：ポンプ{}のprogram numberの指定が間違っています。0~14の数字を指定してください。".format(pump_address))
        if len(program_string) > 128:
            raise ValueError("XLP6000：ポンプ{}のプログラムの文字列が長すぎます。必ず128文字以下にしてください。".format(pump_address))
        command = f"s{program_number}{program_string}"
        self._send_serial_command_sync(pump_address, command)

    def _execute_program_string(self, pump_address:int, program_number:int):
        """
        不揮発性メモリのプログラム文字列を実行する関数。
        :param program_number: プログラム番号（0-14）
        """
        if not (0 <= program_number <= 14):
            raise ValueError("XLP6000：ポンプ{}のprogram numberの指定が間違っています。0~14の数字を指定してください。".format(pump_address))
        command = f"e{program_number}"
        self._send_serial_command_sync(pump_address, command)

    def _load_program_string_async(self, pump_address:int, program_number:int, program_string:str):
        """
        プログラム文字列を不揮発性メモリにロードする関数。
        :param program_number: プログラム番号（0-14）
        :param program_string: プログラム文字列
        """
        if not (0 <= program_number <= 14):
            raise ValueError("XLP6000：ポンプ{}のprogram numberの指定が間違っています。0~14の数字を指定してください。".format(pump_address))
        if len(program_string) > 128:
            raise ValueError("XLP6000：ポンプ{}のプログラムの文字列が長すぎます。必ず128文字以下にしてください。".format(pump_address))
        command = f"s{program_number}{program_string}"
        self._send_serial_command_async(pump_address, command)

    def _execute_program_string_async(self, pump_address:int, program_number:int=1):
        """
        不揮発性メモリのプログラム文字列を実行する関数。
        :param program_number: プログラム番号（0-14）
        """
        if not (0 <= program_number <= 14):
            raise ValueError("XLP6000：ポンプ{}のprogram numberの指定が間違っています。0~14の数字を指定してください。".format(pump_address))
        command = f"e{program_number}"
        self._send_serial_command_async(pump_address, command)

    def _move_valve_to_input(self, pump_address:int):
        """
        ポンプを吸い上げモードに設定する関数
        """
        command = "I"
        self._send_serial_command_sync(pump_address, command)

    def _move_valve_to_output(self, pump_address:int):
        """
        ポンプを吐き出しモードに設定する関数
        """
        command = "O"
        self._send_serial_command_sync(pump_address, command)