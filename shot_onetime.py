def shot_onetime(self):
    # ✅ セミオート1ショット
    start_sig = bytes([0xA5, 0x44, 0xF2, 0x36, 0x5A])  # F2 = Semi-Auto
    completed_sig = [0xA5, 0x44, 0xF9, 0x3D, 0x5A]      # F9 = Completed

    print("Send:", start_sig.hex())
    self._ser.write(start_sig)

    # 完了通知(F9)を待つ
    buffer = []
    while True:
        resp = self._wait_reply_query()
        if resp:
            buffer.extend(resp)

            # 5バイトごとに切り出す
            while len(buffer) >= 5:
                frame = buffer[:5]
                buffer = buffer[5:]

                if frame == completed_sig:
                    print(">>> Completed (F9 detected)")
                    return
                else:
                    print("intermediate:", frame)
        else:
            print("timeout")
            return
