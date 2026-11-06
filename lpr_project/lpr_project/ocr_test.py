"""Thử RIÊNG phần đọc chữ trên biển số (KHÔNG cần best.pt).

    python ocr_test.py --source http://192.168.1.128:8080/video
    python ocr_test.py --source anh.jpg
Đưa biển số vào TRONG KHUNG XANH, rồi nhấn SPACE để đọc. Nhấn q để thoát.
(Khung xanh được xem như biển số, nên biển càng lấp đầy khung càng đọc tốt.)
"""
import argparse
from pathlib import Path

import cv2
import easyocr

import lpr

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--gpu", action="store_true")
args = ap.parse_args()

reader = easyocr.Reader(["en"], gpu=args.gpu)
src = args.source


def read_box(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    text, conf = lpr.read_plate(reader, frame[y1:y2, x1:x2])
    return (x1, y1, x2, y2), text, conf


if Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
    img = cv2.imread(src)
    (x1, y1, x2, y2), text, conf = read_box(img)
    print(f"Đọc được: {lpr.format_plate(text)}  (thô: {text}, tin cậy {conf:.2f})")
    raise SystemExit

cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
last = "SPACE = doc bien so | q = thoat"
while True:
    ok, f = cap.read()
    if not ok:
        print("Mất tín hiệu camera.")
        break
    h, w = f.shape[:2]
    view = f.copy()
    cv2.rectangle(view, (int(w * .1), int(h * .1)), (int(w * .9), int(h * .9)), (0, 255, 0), 3)
    cv2.putText(view, last, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.imshow("OCR test", cv2.resize(view, (1280, int(1280 * h / w))))
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        _, text, conf = read_box(f)
        last = f"{lpr.format_plate(text) or '?'} ({conf:.2f})"
        print(f"Đọc được: {lpr.format_plate(text)}  (thô: {text}, tin cậy {conf:.2f})")
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
