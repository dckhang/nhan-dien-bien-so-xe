"""Quản lý đăng ký biển số xe bằng SQLite.

Dùng như thư viện (import PlateDB) hoặc dòng lệnh:
    # thêm --backend mongo (MongoDB) hoặc --backend excel --excel-file vehicles.xlsx (Excel)
    python db.py add --plate 29A-123.45 --owner "Nguyen Van A" --phone 0901234567 --type "Xe may" --brand Honda --color Do
    python db.py add                      # không có tham số -> hỏi từng trường
    python db.py list
    python db.py search --keyword Nguyen
    python db.py update --plate 29A12345 --phone 0987654321
    python db.py delete --plate 29A12345
    python db.py logs --limit 20
    python db.py export --out vehicles.csv
"""
import argparse
import csv
import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

DB_PATH = "plates.db"
EDITABLE = {"owner_name", "phone", "vehicle_type", "brand", "color", "note"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plate         TEXT NOT NULL UNIQUE,      -- đã chuẩn hóa: chỉ chữ + số, vd 29A12345
    owner_name    TEXT NOT NULL,
    phone         TEXT DEFAULT '',
    vehicle_type  TEXT DEFAULT '',
    brand         TEXT DEFAULT '',
    color         TEXT DEFAULT '',
    note          TEXT DEFAULT '',
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS access_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate       TEXT NOT NULL,               -- biển số đọc được
    vehicle_id  INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    registered  INTEGER NOT NULL,            -- 1: đã đăng ký, 0: chưa
    det_conf    REAL,
    ocr_conf    REAL,
    image_path  TEXT DEFAULT '',
    seen_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_plate ON access_logs(plate);
"""

PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,2}\d?\d{4,5}$")


def normalize_plate(s: str) -> str:
    """'29a-123.45' -> '29A12345'"""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def remove_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt (cv2.putText không vẽ được chữ có dấu)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_vehicle(plate, owner, phone=""):
    """Kiểm tra dữ liệu đăng ký, trả về biển số đã chuẩn hóa. Dùng chung cho SQLite và MongoDB."""
    plate = normalize_plate(plate)
    if not PLATE_RE.match(plate):
        raise ValueError(f"Biển số '{plate}' không đúng định dạng (vd: 29A-123.45, 51F-1234)")
    if not owner.strip():
        raise ValueError("Tên chủ xe không được để trống")
    if phone and not re.fullmatch(r"0\d{9,10}", phone.strip()):
        raise ValueError("Số điện thoại không hợp lệ (vd: 0901234567)")
    return plate


class PlateDB:
    def __init__(self, path: str = DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)  # worker AI chạy ở thread riêng
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- Đăng ký / sửa / xóa ----------
    def register(self, plate, owner, phone="", vehicle_type="", brand="", color="", note=""):
        plate = validate_vehicle(plate, owner, phone)
        try:
            cur = self.conn.execute(
                "INSERT INTO vehicles (plate, owner_name, phone, vehicle_type, brand, color, note, registered_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (plate, owner.strip(), phone.strip(), vehicle_type, brand, color, note, now()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Biển số {plate} đã được đăng ký trước đó")
        return cur.lastrowid

    def update(self, plate, **fields):
        fields = {k: v for k, v in fields.items() if k in EDITABLE and v is not None}
        if not fields:
            raise ValueError("Không có trường nào để cập nhật")
        sets = ", ".join(f"{k}=?" for k in fields)
        cur = self.conn.execute(f"UPDATE vehicles SET {sets} WHERE plate=?",
                                (*fields.values(), normalize_plate(plate)))
        self.conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"Không tìm thấy biển số {plate}")

    def delete(self, plate):
        cur = self.conn.execute("DELETE FROM vehicles WHERE plate=?", (normalize_plate(plate),))
        self.conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"Không tìm thấy biển số {plate}")

    # ---------- Tra cứu ----------
    def get(self, plate):
        return self.conn.execute("SELECT * FROM vehicles WHERE plate=?", (normalize_plate(plate),)).fetchone()

    def find_vehicle(self, plate, max_dist=0):
        """Tìm xe theo biển số đọc từ OCR. max_dist=1 cho phép sai 1 ký tự
        (chỉ chấp nhận nếu có đúng 1 biển gần nhất, tránh nhận nhầm)."""
        plate = normalize_plate(plate)
        row = self.get(plate)
        if row or max_dist == 0 or len(plate) < 7:
            return row
        cands = []
        for r in self.conn.execute("SELECT * FROM vehicles"):
            d = edit_distance(plate, r["plate"])
            if d <= max_dist:
                cands.append((d, r))
        cands.sort(key=lambda x: x[0])
        if len(cands) == 1 or (len(cands) > 1 and cands[0][0] < cands[1][0]):
            return cands[0][1]
        return None

    def list_vehicles(self, keyword=None):
        if keyword:
            k = f"%{keyword}%"
            return self.conn.execute(
                "SELECT * FROM vehicles WHERE plate LIKE ? OR owner_name LIKE ? OR phone LIKE ? "
                "OR brand LIKE ? ORDER BY registered_at DESC", (k, k, k, k)).fetchall()
        return self.conn.execute("SELECT * FROM vehicles ORDER BY registered_at DESC").fetchall()

    # ---------- Nhật ký ----------
    def log_access(self, plate, vehicle_id=None, det_conf=None, ocr_conf=None, image_path=""):
        self.conn.execute(
            "INSERT INTO access_logs (plate, vehicle_id, registered, det_conf, ocr_conf, image_path, seen_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (normalize_plate(plate), vehicle_id, int(vehicle_id is not None), det_conf, ocr_conf, image_path, now()))
        self.conn.commit()

    def list_logs(self, limit=50):
        return self.conn.execute(
            "SELECT l.seen_at, l.plate, l.registered, v.owner_name, l.ocr_conf "
            "FROM access_logs l LEFT JOIN vehicles v ON v.id = l.vehicle_id "
            "ORDER BY l.id DESC LIMIT ?", (limit,)).fetchall()

    def export_csv(self, out):
        rows = self.list_vehicles()
        with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig để Excel đọc đúng tiếng Việt
            w = csv.writer(f)
            w.writerow(rows[0].keys() if rows else ["plate"])
            w.writerows([tuple(r) for r in rows])
        return len(rows)


def add_db_args(ap):
    """Thêm các tham số chọn CSDL vào argparse (dùng chung cho db.py và lpr.py)."""
    ap.add_argument("--backend", choices=["sqlite", "mongo", "excel"], default="sqlite", help="loại CSDL")
    ap.add_argument("--db", default=DB_PATH, help="[sqlite] đường dẫn file .db")
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017", help="[mongo] chuỗi kết nối")
    ap.add_argument("--mongo-db", default="lpr", help="[mongo] tên database")
    ap.add_argument("--excel-file", default="vehicles.xlsx", help="[excel] đường dẫn file .xlsx")


def open_db(args):
    if args.backend == "mongo":
        from mongo_db import MongoPlateDB  # import muộn: chỉ cần pymongo khi dùng MongoDB
        return MongoPlateDB(args.mongo_uri, args.mongo_db)
    if args.backend == "excel":
        from excel_db import ExcelPlateDB
        return ExcelPlateDB(args.excel_file)
    return PlateDB(args.db)


def pretty(plate: str) -> str:
    """29A12345 -> 29A-123.45"""
    if not PLATE_RE.match(plate):
        return plate
    n = 4 if len(plate) == 7 else 5
    head, tail = plate[:-n], plate[-n:]
    return f"{head}-{tail[:3]}.{tail[3:]}" if n == 5 else f"{head}-{tail}"


# ---------- CLI ----------
def _ask(prompt, default=""):
    v = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
    return v or default


def main():
    ap = argparse.ArgumentParser(description="Quản lý biển số xe đã đăng ký")
    add_db_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="đăng ký xe mới")
    a.add_argument("--plate"); a.add_argument("--owner"); a.add_argument("--phone", default=None)
    a.add_argument("--type", dest="vehicle_type", default=None)
    a.add_argument("--brand", default=None); a.add_argument("--color", default=None); a.add_argument("--note", default=None)

    u = sub.add_parser("update", help="sửa thông tin")
    u.add_argument("--plate", required=True)
    u.add_argument("--owner", dest="owner_name"); u.add_argument("--phone")
    u.add_argument("--type", dest="vehicle_type"); u.add_argument("--brand")
    u.add_argument("--color"); u.add_argument("--note")

    d = sub.add_parser("delete", help="xóa xe"); d.add_argument("--plate", required=True)
    s = sub.add_parser("search", help="tìm kiếm"); s.add_argument("--keyword", required=True)
    sub.add_parser("list", help="liệt kê tất cả xe")
    lg = sub.add_parser("logs", help="xem nhật ký nhận diện"); lg.add_argument("--limit", type=int, default=50)
    ex = sub.add_parser("export", help="xuất CSV"); ex.add_argument("--out", default="vehicles.csv")

    args = ap.parse_args()
    db = open_db(args)

    try:
        if args.cmd == "add":
            # thiếu tham số nào thì hỏi tham số đó
            plate = args.plate or _ask("Biển số (vd 29A-123.45)")
            owner = args.owner or _ask("Họ tên chủ xe")
            phone = args.phone if args.phone is not None else _ask("Số điện thoại")
            vtype = args.vehicle_type if args.vehicle_type is not None else _ask("Loại xe (xe máy/ô tô...)")
            brand = args.brand if args.brand is not None else _ask("Hãng xe")
            color = args.color if args.color is not None else _ask("Màu xe")
            note = args.note if args.note is not None else _ask("Ghi chú")
            db.register(plate, owner, phone, vtype, brand, color, note)
            print(f"Đã đăng ký {pretty(normalize_plate(plate))} - {owner}")
        elif args.cmd == "update":
            db.update(args.plate, owner_name=args.owner_name, phone=args.phone, vehicle_type=args.vehicle_type,
                      brand=args.brand, color=args.color, note=args.note)
            print("Đã cập nhật.")
        elif args.cmd == "delete":
            db.delete(args.plate)
            print("Đã xóa.")
        elif args.cmd in ("list", "search"):
            rows = db.list_vehicles(getattr(args, "keyword", None))
            if not rows:
                print("Không có dữ liệu.")
            for r in rows:
                print(f"{pretty(r['plate']):<14} {r['owner_name']:<22} {r['phone']:<12} "
                      f"{r['vehicle_type']:<8} {r['brand']:<10} {r['color']:<8} {r['registered_at']}")
        elif args.cmd == "logs":
            for r in db.list_logs(args.limit):
                status = f"DA DANG KY ({r['owner_name']})" if r["registered"] else "CHUA DANG KY"
                print(f"{r['seen_at']}  {pretty(r['plate']):<14} {status}")
        elif args.cmd == "export":
            print(f"Đã xuất {db.export_csv(args.out)} xe ra {args.out}")
    except ValueError as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
