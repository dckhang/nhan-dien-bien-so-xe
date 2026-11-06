# Nhận diện biển số xe (YOLO + EasyOCR + CSDL)

## Cấu trúc
| File | Chức năng |
|---|---|
| `train.py` | Huấn luyện YOLO phát hiện biển số |
| `data.yaml` | Cấu hình dataset (định dạng YOLO) |
| `lpr.py` | Nhận diện biển số (ảnh / video / webcam / camera điện thoại) + đối chiếu CSDL |
| `db.py` | CSDL SQLite + CLI quản lý xe đăng ký |
| `mongo_db.py` | CSDL MongoDB (cùng giao diện với `db.py`) |
| `excel_db.py` | CSDL Excel (`vehicles.xlsx`) - đăng ký bằng cách gõ thẳng vào Excel |
| `vehicles.xlsx` | File Excel mẫu (có sheet hướng dẫn) |
| `app.py` | Giao diện web Streamlit: form đăng ký, danh sách, nhật ký |
| `check_camera.py` | Kiểm tra kết nối camera điện thoại |
| `requirements.txt` | Thư viện cần cài |

## Cài đặt
```bash
pip install -r requirements.txt
```

## Quy trình
```bash
# 1. Train (dataset đặt trong datasets/license_plate/, xem data.yaml)
python train.py --data data.yaml --model yolov8n.pt --epochs 100 --device 0

# 2. Đăng ký xe qua web
streamlit run app.py
#    hoặc dòng lệnh: python db.py add

# 3. Kiểm tra camera điện thoại (cùng Wi-Fi), rồi nhận diện
python check_camera.py --source http://192.168.1.10:8080/video
python lpr.py --weights runs/detect/plate_det/weights/best.pt --source http://192.168.1.10:8080/video

# Ảnh / video
python lpr.py --weights best.pt --source anh.jpg --save
python lpr.py --weights best.pt --source video.mp4 --save

# Đổi CSDL (mặc định SQLite):
#   MongoDB: --backend mongo [--mongo-uri ...]
#   Excel  : --backend excel --excel-file vehicles.xlsx   (áp dụng cho db.py và lpr.py)
```
Xem log: `python db.py logs` hoặc tab "Nhật ký" trong web. Xuất CSV: `python db.py export`.
