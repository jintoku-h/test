def shot_onetime(self):

    start_sig = bytes([0xA5, 0x44, 0xF2, 0x36, 0x5A])
    completed = [0xA5, 0x44, 0xF9, 0x3D, 0x5A]

    print("Send:", start_sig.hex())
    self._ser.write(start_sig)

    buffer = bytearray()

    t0 = time.time()
    timeout = 10

    while time.time() - t0 < timeout:

        if self._ser.in_waiting:
            data = self._ser.read(self._ser.in_waiting)
            buffer.extend(data)

            # バイトストリーム中から 5バイトフレームをスキャン
            for i in range(len(buffer) - 4):
                frame = buffer[i:i+5]

                if list(frame) == completed:
                    print(">>> Completed(F9) detected")
                    return True

        time.sleep(0.01)

    print("Timeout (F9 not detected)")
    return False
