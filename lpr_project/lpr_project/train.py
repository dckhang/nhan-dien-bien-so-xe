"""Huấn luyện YOLO phát hiện biển số xe.

Chạy:
    python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
Kết quả: runs/detect/<name>/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt hoặc yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 cho GPU, cpu cho CPU")
    ap.add_argument("--name", default="plate_det")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,       # dừng sớm nếu không cải thiện
        # augmentation phù hợp biển số (không lật ngang vì làm sai chữ)
        fliplr=0.0,
        degrees=5,
        perspective=0.0005,
        hsv_v=0.4,
        mosaic=1.0,
    )

    # Đánh giá trên tập val: in ra Precision, Recall, mAP50, mAP50-95
    metrics = model.val()
    print(f"Precision: {metrics.box.mp:.3f} | Recall: {metrics.box.mr:.3f} | "
          f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
