import time, serial
import serial.serialutil

def _ensure_open(self, port="COM8"):
    """必要に応じて再オープン（__init__で1度呼ぶのが基本）"""
    if getattr(self, "_ser", None) and self._ser.is_open:
        return
    self._ser = serial.Serial(
        port=port,
        baudrate=19200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.2,
        write_timeout=1.0,
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
        exclusive=True,      # 他プロセスからの掴みを防ぐ
        inter_byte_timeout=None,
    )
    # DTR/RTSの状態を固定（機種によってはリセット回避に必須）
    self._ser.setDTR(True)
    self._ser.setRTS(False)
    self._ser.reset_input_buffer()
    self._ser.reset_output_buffer()

def shot_onetime(self, wait_sec: float = 4.0):
    """
    F9が返らないFW3.7個体向け。
    - ポートは開きっぱなし（withは使わない）
    - F2(0xA5 44 F2 36 5A) 送出後、一定時間は受信を捌いて
      '通信が早く終わる'事による切断を防ぐ
    """
    self._ensure_open()  # 念のため

    start_sig = bytes([0xA5, 0x44, 0xF2, 0x36, 0x5A])  # Semi-auto 1shot（あなたの個体で受理）

    try:
        # 送出
        self._ser.write(start_sig)
        self._ser.flush()  # 即送信

        # 送信直後のエコーを軽く読む（なくてもOK）
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if self._ser.in_waiting:
                _ = self._ser.read(self._ser.in_waiting)

        # 完了(F9)は来ない個体なので、装置のサイクル時間だけ
        # "受信を捌き続けて" ポートを活性状態に保つ
        t0 = time.time()
        while time.time() - t0 < wait_sec:
            if self._ser.in_waiting:
                _ = self._ser.read(self._ser.in_waiting)
            time.sleep(0.01)  # 過度にCPUを使わず、かつポーリングを継続

        return True

    except (serial.SerialException, serial.serialutil.SerialException) as e:
        # 途中切断時の簡易リカバリ
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        self._ensure_open()
        # 1回だけリトライ（必要なら回数を増やす）
        try:
            self._ser.write(start_sig)
            t0 = time.time()
            while time.time() - t0 < wait_sec:
                if self._ser.in_waiting:
                    _ = self._ser.read(self._ser.in_waiting)
                time.sleep(0.01)
            return True
        except Exception:
            return False
