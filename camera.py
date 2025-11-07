import cv2

def list_available_cameras(max_cameras=10):
    print("使用可能なカメラをスキャン中...")
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"カメラが見つかりました：デバイス番号 {index}")
            available_cameras.append(index)
        cap.release()
    if not available_cameras:
        print("使用可能なカメラが見つかりませんでした。")
    return available_cameras

list_available_cameras()
