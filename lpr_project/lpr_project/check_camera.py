"""Kiểm tra kết nối camera điện thoại TRƯỚC khi chạy nhận diện.

    python check_camera.py --source http://192.168.1.128:8080/video
    python check_camera.py --source 1            # DroidCam/Iriun/Camo (điện thoại giả làm webcam)
Hiện hình + FPS + độ phân giải; nhấn 'q' để thoát.
"""
import argparse
import time

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
src = ap.parse_args().source
cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
if not cap.isOpened():
    raise SystemExit("Không mở được camera. Kiểm tra IP:cổng, cùng Wi-Fi, app đã bật server chưa.")
n, t0, fps = 0, time.time(), 0.0
while True:
    ok, f = cap.read()
    if not ok:
        print("Mất tín hiệu.")
        break
    n += 1
    if time.time() - t0 >= 1:
        fps, n, t0 = n / (time.time() - t0), 0, time.time()
    h, w = f.shape[:2]
    cv2.putText(f, f"{w}x{h}  {fps:.0f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("check camera", f)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
