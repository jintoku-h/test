def shot_onetime_with_final_vib(self,
                                dummy_stop_ms: int = 10,
                                dummy_speed_level: int = 2,   # 0:Fast, 1:Normal, 2:Slow
                                settle_sec: float = 0.3,
                                cycle_sec_main: float = 3.5,
                                cycle_sec_dummy: float = 2.0):
    """
    1) 通常1ショット (F2)
    2) ダミー1ショットで「後振動だけ」を事実上付与
       - 実行前に motor speed=Slow, stop pos=小 へ一時変更
       - 実行後に元へ戻す

    dummy_stop_ms: ダミー時の停止位置(ms)。小さいほど粉が出にくい（0-150）
    dummy_speed_level: 2=Slow を推奨
    cycle_sec_main: 通常ショットに必要な待ち時間（装置に合わせて調整）
    cycle_sec_dummy: ダミーショットの待ち時間（短め）
    """
    # 1) 現在設定を読む（任意：復元用）
    #   0x51 応答は [0x51, Pause, MotorSpeed, VibTime, StopPos]
    try:
        self.status_query()  # 画面表示も兼ねる
    except Exception:
        pass

    # 2) 通常ショット
    print("[Main] 1shot start (F2)")
    self.shot_onetime()
    # F9が返らない個体想定なので時間で待つ
    time.sleep(cycle_sec_main)

    # 3) 後振動用のダミー設定に一時変更
    print("[Dummy] apply minimal-dispense settings")
    # VibTime 自体はそのままでOK（開始前/終了後に効く）
    self.set_motor_speed(dummy_speed_level)       # Slow = 0x7F
    self.set_motor_stop_time(dummy_stop_ms)       # できるだけ小さく
    # Pause(間隔)を0秒化したい場合
    try:
        self.set_wait_time(0)
    except Exception:
        pass
    time.sleep(settle_sec)  # EEPROM書込の落ち着き待ち

    # 4) ダミーショット（振動だけ欲しい）
    print("[Dummy] 1shot start (F2) for final vibration")
    self.shot_onetime()
    time.sleep(cycle_sec_dummy)

    # 5) もとに戻す（必要な値に合わせて再設定）
    print("[Restore] restore preferred settings")
    # 例：Normal/既定Stopに戻す。プロジェクト標準に合わせて調整してください。
    self.set_motor_speed(1)          # Normal(0x3F)
    self.set_motor_stop_time(50)     # 例: 50ms
    # Pause も戻す
    try:
        self.set_wait_time(2)        # 例: 2秒
    except Exception:
        pass
    time.sleep(settle_sec)
    print("[Done] appended final vibration via dummy cycle")
