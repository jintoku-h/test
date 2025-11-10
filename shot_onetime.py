def shot_onetime(self):
    """
    SDB-1（Firmware 3.7）用 正しい1ショット（セミオート）
    Start = F2（セミオート1回）
    Completed = F9（完了通知）
    """
    # ✅ セミオート（1 shot）
    start_sig = bytes([0xA5, 0x44, 0xF2, 0x36, 0x5A])   # F2 = Semi-auto
    completed_sig = [0xA5, 0x44, 0xF9, 0x3D, 0x5A]      # F9 = Completed

    print(f"送信しました: {start_sig.hex()}")
    super()._send_data(start_sig)

    buffer = bytearray()

    while True:
        # 応答を待つ
        data = super()._wait_reply_query()
        if not data:
            print("受信なし／タイムアウトで終了")
            return

        buffer.extend(data)
        print(f"受信: {data.hex()} (len={len(data)})")

        # 5バイトフレームを順次解析
        while len(buffer) >= 5:
            frame = list(buffer[:5])
            del buffer[:5]

            # ✅ 完了通知（F9）
            if frame == completed_sig:
                print(">>> 完了信号(F9)検出 → 後振動開始")
                # 振動時間分（装置内部設定による）待機
                time.sleep(0.2)  # 最低限の待機
                return

            # 中間フレーム
            else:
                print(f"中間信号: {[hex(b) for b in frame]}")
