"""Nhận diện biển số xe: YOLO (phát hiện) + EasyOCR (đọc ký tự) + đối chiếu CSDL.

Ví dụ:
    python lpr.py --weights runs/detect/plate_det/weights/best.pt --source anh.jpg
    python lpr.py --weights best.pt --source video.mp4 --save
    python lpr.py --weights best.pt --source 0            # webcam
"""
import argparse
import os
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from db import add_db_args, normalize_plate, open_db, remove_accents

# ---------- Hậu xử lý theo định dạng biển số Việt Nam ----------
# Ví dụ: 29A-12345, 51F-123.45, 59-X1 123.45 (xe máy: 2 dòng)
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d?\d{4,5}$")


def clean_text(text: str) -> str:
    """Giữ lại chữ + số, viết hoa."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def fix_vn_plate(s: str) -> str:
    """Sửa các nhầm lẫn thường gặp theo vị trí: 2 ký tự đầu là số (mã tỉnh),
    ký tự thứ 3 là chữ (seri), phần còn lại là số."""
    if len(s) < 7:
        return s
    chars = list(s)
    for i in (0, 1):
        chars[i] = TO_DIGIT.get(chars[i], chars[i])
    chars[2] = TO_ALPHA.get(chars[2], chars[2])
    # ký tự thứ 4 có thể là chữ (vd 29MD1) hoặc số; các ký tự cuối chắc chắn là số
    for i in range(4, len(chars)):
        chars[i] = TO_DIGIT.get(chars[i], chars[i])
    return "".join(chars)


def format_plate(s: str) -> str:
    """29A12345 -> 29A-123.45 (dễ đọc hơn)."""
    if not PLATE_RE.match(s):
        return s
    n = 4 if len(s) == 7 else 5  # biển cũ 4 số, biển mới 5 số
    head, tail = s[:-n], s[-n:]
    return f"{head}-{tail[:3]}.{tail[3:]}" if n == 5 else f"{head}-{tail}"


# ---------- Tiền xử lý ảnh biển số ----------
def preprocess(plate_bgr: np.ndarray) -> np.ndarray:
    h, w = plate_bgr.shape[:2]
    if h < 200:        # biển quá nhỏ -> phóng lên
        img = cv2.resize(plate_bgr, None, fx=200 / max(h, 1), fy=200 / max(h, 1), interpolation=cv2.INTER_CUBIC)
    elif h > 360:      # biển quá lớn (camera 1080p) -> thu nhỏ để OCR trên CPU nhanh hơn
        img = cv2.resize(plate_bgr, None, fx=360 / h, fy=360 / h, interpolation=cv2.INTER_AREA)
    else:
        img = plate_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def read_plate(reader: easyocr.Reader, plate_bgr: np.ndarray):
    """Trả về (text, confidence). Tự nhận biết biển 1 dòng / 2 dòng theo tỉ lệ khung."""
    h, w = plate_bgr.shape[:2]
    img = preprocess(plate_bgr)
    allow = "ABCDEFGHKLMNPRSTUVXYZ0123456789"

    def ocr(im):
        res = reader.readtext(im, allowlist=allow, detail=1, paragraph=False)
        res.sort(key=lambda r: r[0][0][0])  # sắp xếp trái -> phải
        txt = "".join(r[1] for r in res)
        conf = float(np.mean([r[2] for r in res])) if res else 0.0
        return txt, conf

    if w / h < 2.5:  # biển vuông (xe máy) -> cắt 2 dòng
        mid = img.shape[0] // 2
        t1, c1 = ocr(img[:mid, :])
        t2, c2 = ocr(img[mid:, :])
        text, conf = t1 + t2, (c1 + c2) / 2
    else:
        text, conf = ocr(img)

    text = fix_vn_plate(clean_text(text))
    return text, conf


# ---------- Pipeline ----------
GREEN, RED, ORANGE, YELLOW = (0, 200, 0), (0, 0, 255), (0, 165, 255), (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BACKEND_NAMES = {"sqlite": "SQLite", "mongo": "MongoDB", "excel": "Excel"}


def scale_of(frame):
    """Hệ số phóng chữ/nét vẽ theo độ rộng ảnh (ảnh 1920px cần chữ to hơn ảnh 640px)."""
    return max(1.0, frame.shape[1] / 1000)


def put_text_outline(frame, text, org, scale, color, thick):
    """Chữ có viền đen để dễ đọc trên mọi nền."""
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_label(frame, box, label, color):
    """Khung quanh biển số + nhãn ngay phía trên khung."""
    s = scale_of(frame)
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(3 * s)))
    scale, thick = 0.7 * s, max(2, int(2 * s))
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
    top = max(0, y1 - th - 12)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + 12), color, -1)
    cv2.putText(frame, label, (x1 + 4, top + th + 4), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


class Banner:
    """Thông báo hiện ở góc trên bên trái ảnh camera trong vài giây (vd: DA LUU [Excel]: 99E1-222.68)."""

    def __init__(self, seconds=4.0):
        self.seconds, self.lines, self.until = seconds, [], 0.0
        self.lock = threading.Lock()

    def show(self, lines):
        with self.lock:
            self.lines, self.until = lines, time.time() + self.seconds

    def draw(self, frame, y=None):
        """Vẽ các dòng thông báo; trả về toạ độ y để vẽ dòng tiếp theo."""
        s = scale_of(frame)
        y = int(45 * s) if y is None else y
        with self.lock:
            lines, until = list(self.lines), self.until
        if time.time() > until:
            return y
        for text, color in lines:
            put_text_outline(frame, text, (int(20 * s), y), 1.0 * s, color, max(2, int(2 * s)))
            y += int(48 * s)
        return y


def draw_notifications(frame, state):
    """Dòng trạng thái (luôn hiện khi thấy biển) + thông báo sự kiện 'DA LUU' bên dưới."""
    y = state.status.draw(frame)
    state.banner.draw(frame, y)


class PlateVoter:
    """Bỏ phiếu qua N lần đọc gần nhất: một biển chỉ được xác nhận khi xuất hiện >= min_votes lần.
    Giúp loại các lần OCR đọc nhiễu theo từng khung hình."""

    def __init__(self, window=7, min_votes=3, timeout=6.0):
        self.buf, self.min_votes, self.timeout, self.last = deque(maxlen=window), min_votes, timeout, 0.0
        self.best, self.n = None, 0   # ứng viên dẫn đầu và số phiếu (để hiện tiến độ)

    def add(self, plate):
        now_t = time.time()
        if now_t - self.last > self.timeout:   # lâu không thấy biển -> bỏ phiếu cũ
            self.buf.clear()
        self.last = now_t
        self.buf.append(plate)
        self.best, self.n = Counter(self.buf).most_common(1)[0]
        return self.best if self.n >= self.min_votes else None


def banner_lines(args, pdb, plate, vehicle):
    if pdb is None:
        return [(f"DOC DUOC: {format_plate(plate)}", YELLOW)]
    name = BACKEND_NAMES.get(args.backend, args.backend)
    shown = vehicle["plate"] if vehicle else normalize_plate(plate)
    return [(f"DA LUU [{name}]: {format_plate(shown)}", YELLOW)]


def status_line(text, ocr_conf, valid, stable, vehicle, pdb, state):
    """Dòng trạng thái hiện liên tục trên ảnh camera cho mỗi biển đang thấy."""
    if not text:
        return ("DANG DOC BIEN SO...", ORANGE)
    if not valid:   # sai định dạng hoặc OCR chưa đủ tin cậy
        return (f"DANG DOC: {format_plate(text)} (tin cay {ocr_conf:.2f})", ORANGE)
    if not stable:  # đúng định dạng, đang gom phiếu
        v = state.voter
        return (f"DANG XAC NHAN: {format_plate(v.best)} ({v.n}/{v.min_votes})", ORANGE)
    if pdb is None:
        return (f"BIEN SO: {format_plate(stable)}", GREEN)
    if vehicle:
        return (f"DA DANG KY: {remove_accents(vehicle['owner_name'])} | {format_plate(vehicle['plate'])}", GREEN)
    return (f"CHUA DANG KY: {format_plate(stable)}", RED)


def process_frame(frame, detector, reader, args, pdb=None, last_seen=None, draw=True, state=None):
    """Phát hiện + đọc biển + bỏ phiếu + đối chiếu CSDL + ghi log + hiện thông báo.
    state: SimpleNamespace(voter, banner). last_seen: dict chống ghi log trùng liên tục."""
    results = detector.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    out, status = [], []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])
        pad = 4
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        text, ocr_conf = read_plate(reader, crop)
        valid = bool(PLATE_RE.match(text)) and ocr_conf >= args.ocr_conf

        stable = None   # biển đã được xác nhận qua bỏ phiếu
        if valid:
            stable = state.voter.add(text) if state else text
        vehicle = None
        if stable and pdb is not None:
            vehicle = pdb.find_vehicle(stable, max_dist=args.fuzzy)

        # nhãn + màu: xanh = đã đăng ký, đỏ = chưa đăng ký, cam = đang xác nhận / đọc chưa chắc
        if not text:
            label, color = "?", ORANGE
        elif not stable:
            label, color = f"{format_plate(text)}?", ORANGE
        elif pdb is None:
            label, color = format_plate(stable), GREEN
        elif vehicle:
            label = f"{format_plate(vehicle['plate'])} | {remove_accents(vehicle['owner_name'])}"
            color = GREEN
        else:
            label, color = f"{format_plate(stable)} | CHUA DANG KY", RED
        if draw:
            draw_label(frame, (x1, y1, x2, y2), label, color)
        status.append(status_line(text, ocr_conf, valid, stable, vehicle, pdb, state))

        # sự kiện xác nhận: ghi nhật ký + hiện thông báo (mỗi biển 1 lần / --cooldown giây)
        if stable:
            key = vehicle["plate"] if vehicle else normalize_plate(stable)
            t = time.time()
            if last_seen is not None and t - last_seen.get(key, 0) >= args.cooldown:
                last_seen[key] = t
                snap = ""
                if args.save:
                    snap_dir = Path("output/snapshots")
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap = str(snap_dir / f"{key}_{int(t)}.jpg")
                    cv2.imwrite(snap, crop)
                if pdb is not None:
                    pdb.log_access(key, vehicle["id"] if vehicle else None, det_conf, ocr_conf, snap)
                if state:
                    state.banner.show(banner_lines(args, pdb, stable, vehicle))

        out.append({"bbox": (x1, y1, x2, y2), "label": label, "color": color,
                    "text": stable or text, "ocr_conf": ocr_conf,
                    "registered": vehicle is not None,
                    "owner": vehicle["owner_name"] if vehicle else None})
    if state and status:
        state.status.show(status[:2])
    return frame, out


class FrameGrabber(threading.Thread):
    """Đọc camera ở luồng riêng và chỉ giữ khung hình MỚI NHẤT -> không bị trễ tích lũy."""

    def __init__(self, src):
        super().__init__(daemon=True)
        self.src, self.frame, self.seq = src, None, 0
        self.lock, self.stopped, self.cap = threading.Lock(), False, None
        if not self._open():
            raise SystemExit(f"Không mở được nguồn camera: {src}\n"
                             "Kiểm tra: điện thoại và máy tính cùng Wi-Fi, đúng IP:cổng, app đã bấm Start server.")

    def _open(self):
        if isinstance(self.src, str) and self.src.startswith("rtsp"):
            # giảm độ trễ cho RTSP (TCP ổn định hơn UDP trên Wi-Fi)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.cap = cv2.VideoCapture(self.src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return self.cap.isOpened()

    def run(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                print("Mất kết nối camera, đang thử kết nối lại...")
                with self.lock:
                    self.frame = None
                self.cap.release()
                while not self.stopped and not self._open():
                    time.sleep(1)
                continue
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def read(self):
        with self.lock:
            return (None, self.seq) if self.frame is None else (self.frame.copy(), self.seq)

    def stop(self):
        self.stopped = True
        self.join(timeout=3)             # đợi luồng đọc dừng hẳn rồi mới đóng camera (tránh crash khi thoát)
        if self.cap is not None and not self.is_alive():
            self.cap.release()


class InferenceWorker(threading.Thread):
    """Chạy YOLO + OCR trên khung hình mới nhất, độc lập với tốc độ hiển thị."""

    def __init__(self, grabber, fn):
        super().__init__(daemon=True)
        self.grabber, self.fn, self.stopped = grabber, fn, False
        self.lock, self.dets, self.stamp, self.fps = threading.Lock(), [], 0.0, 0.0

    def run(self):
        last_seq = -1
        while not self.stopped:
            frame, seq = self.grabber.read()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            t = time.time()
            _, dets = self.fn(frame)
            with self.lock:
                self.dets, self.stamp = dets, time.time()
                self.fps = 1.0 / max(time.time() - t, 1e-6)

    def latest(self):
        with self.lock:
            return self.dets, time.time() - self.stamp


def run_live(args, cam_src, detector, reader, pdb, last_seen, out_dir, state):
    grabber = FrameGrabber(cam_src)
    worker = InferenceWorker(
        grabber, lambda f: process_frame(f, detector, reader, args, pdb, last_seen, draw=False, state=state))
    grabber.start()
    worker.start()
    show, writer = not args.no_show, None
    t_start = t_fps = time.time()
    n, disp_fps = 0, 0.0
    print("Đang chạy. Nhấn 'q' trong cửa sổ video (hoặc Ctrl+C) để dừng.")
    try:
        while True:
            frame, _ = grabber.read()
            if frame is None:
                time.sleep(0.02)
                continue
            dets, age = worker.latest()
            if age < 1.0:  # chỉ vẽ kết quả còn "tươi"
                for d in dets:
                    draw_label(frame, d["bbox"], d["label"], d["color"])
            draw_notifications(frame, state)
            n += 1
            if time.time() - t_fps >= 1:
                disp_fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
            sc = scale_of(frame)
            put_text_outline(frame, f"cam {disp_fps:.0f} fps | AI {worker.fps:.1f} fps",
                             (10, frame.shape[0] - int(15 * sc)), 0.6 * sc, (255, 255, 0), 1)
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "result.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("LPR - camera dien thoai", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)
            if args.duration and time.time() - t_start > args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        worker.stopped = True
        grabber.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def describe(dets):
    if not dets:
        return "không thấy biển số"
    parts = []
    for d in dets:
        status = f"đã đăng ký: {d['owner']}" if d["registered"] else "chưa đăng ký"
        parts.append(f"{format_plate(d['text'])} ({status})")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="file best.pt sau khi train")
    ap.add_argument("--source", required=True, help="ảnh / video / thư mục ảnh / 0 (webcam)")
    ap.add_argument("--conf", type=float, default=0.4, help="ngưỡng phát hiện YOLO")
    ap.add_argument("--ocr-conf", type=float, default=0.3, help="ngưỡng tin cậy OCR")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--gpu", action="store_true", help="dùng GPU cho EasyOCR")
    ap.add_argument("--save", action="store_true", help="lưu kết quả vào thư mục output/")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ kết quả")
    ap.add_argument("--no-show", action="store_true", help="camera trực tiếp: không mở cửa sổ (chạy nền)")
    ap.add_argument("--duration", type=int, default=0, help="camera trực tiếp: tự dừng sau N giây (0 = chạy mãi)")
    add_db_args(ap)
    ap.add_argument("--no-db", action="store_true", help="chỉ nhận diện, không đối chiếu/ghi CSDL")
    ap.add_argument("--fuzzy", type=int, default=1, help="cho phép sai tối đa N ký tự khi đối chiếu (0 = khớp tuyệt đối)")
    ap.add_argument("--cooldown", type=int, default=30, help="số giây tối thiểu giữa 2 lần ghi log cùng 1 biển")
    ap.add_argument("--vote-window", type=int, default=7, help="số lần đọc gần nhất dùng để bỏ phiếu")
    ap.add_argument("--min-votes", type=int, default=3, help="số lần đọc giống nhau tối thiểu để xác nhận biển")
    ap.add_argument("--banner-sec", type=float, default=4.0, help="số giây hiện thông báo trên ảnh camera")
    args = ap.parse_args()

    detector = YOLO(args.weights)
    reader = easyocr.Reader(["en"], gpu=args.gpu)
    pdb = None if args.no_db else open_db(args)
    last_seen = {}
    state = SimpleNamespace(voter=PlateVoter(args.vote_window, args.min_votes), banner=Banner(args.banner_sec),
                          status=Banner(1.0))
    out_dir = Path("output")
    if args.save:
        out_dir.mkdir(exist_ok=True)

    src = args.source
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    is_live = src.isdigit() or src.lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))

    # --- Webcam / camera điện thoại (luồng mạng): đọc đa luồng để không bị trễ ---
    if is_live:
        run_live(args, int(src) if src.isdigit() else src, detector, reader, pdb, last_seen, out_dir, state)
        return

    # --- Ảnh đơn hoặc thư mục ảnh ---
    p = Path(src)
    if p.is_dir() or p.suffix.lower() in img_exts:
        state.voter = PlateVoter(1, 1)   # ảnh tĩnh: chỉ có 1 lần đọc nên không cần bỏ phiếu
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in img_exts) if p.is_dir() else [p]
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            frame, dets = process_frame(frame, detector, reader, args, pdb, last_seen, state=state)
            draw_notifications(frame, state)
            print(f"{f.name}: {describe(dets)}")
            if args.save:
                cv2.imwrite(str(out_dir / f.name), frame)
            if args.show:
                cv2.imshow("LPR", frame)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
        return

    # --- File video ---
    cap = cv2.VideoCapture(src)
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(out_dir / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame, dets = process_frame(frame, detector, reader, args, pdb, last_seen, state=state)
        draw_notifications(frame, state)
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("LPR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()