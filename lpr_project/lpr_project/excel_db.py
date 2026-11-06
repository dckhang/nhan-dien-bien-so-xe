"""Backend Excel cho hệ thống đăng ký biển số (cùng giao diện với PlateDB / MongoPlateDB).

- vehicles.xlsx : danh sách xe đăng ký. Bạn có thể MỞ BẰNG EXCEL VÀ GÕ TRỰC TIẾP mỗi xe một dòng
                  (nhớ Ctrl+S). Chương trình tự đọc lại khi file thay đổi.
- vehicles_logs.csv : nhật ký nhận diện (ghi riêng để không đụng file Excel đang mở).
"""
import csv
import threading
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from db import EDITABLE, PLATE_RE, edit_distance, normalize_plate, now, pretty, validate_vehicle

HEADERS = ["Biển số", "Chủ xe", "Số điện thoại", "Loại xe", "Hãng xe", "Màu xe", "Ghi chú", "Ngày đăng ký"]
KEYS = ["plate", "owner_name", "phone", "vehicle_type", "brand", "color", "note", "registered_at"]
WIDTHS = [16, 26, 16, 14, 14, 12, 30, 20]
VEHICLE_TYPES = ["Xe máy", "Ô tô", "Xe tải", "Xe đạp điện", "Khác"]
FORMATTED_ROWS = 2000
LOG_FIELDS = ["seen_at", "plate", "registered", "owner_name", "det_conf", "ocr_conf", "image_path"]
LOCKED_MSG = "Không ghi được file Excel. Hãy đóng file đó trong Excel (hoặc Save rồi đóng) rồi thử lại."


def _s(v):
    """Giá trị ô Excel -> chuỗi sạch."""
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


class ExcelPlateDB:
    def __init__(self, path="vehicles.xlsx", log_path=None):
        self.path = Path(path)
        self.log_path = Path(log_path) if log_path else self.path.with_name(self.path.stem + "_logs.csv")
        self._lock = threading.RLock()
        self._cache, self._sig, self._warned = None, None, False
        if not self.path.exists():
            self._create()

    # ---------- Tạo file mẫu ----------
    def _create(self):
        wb = Workbook()
        ws = wb.active
        ws.title = "Xe đăng ký"
        ws.append(HEADERS)
        fill = PatternFill("solid", fgColor="1F4E78")
        for i, w in enumerate(WIDTHS, 1):
            c = ws.cell(1, i)
            c.font, c.fill = Font(bold=True, color="FFFFFF"), fill
            c.alignment = Alignment(horizontal="center", vertical="center")
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
        for r in range(2, FORMATTED_ROWS + 2):      # cột Biển số + SĐT là dạng Text để Excel không bỏ số 0 đầu
            ws.cell(r, 1).number_format = "@"
            ws.cell(r, 3).number_format = "@"
        dv = DataValidation(type="list", formula1='"' + ",".join(VEHICLE_TYPES) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"D2:D{FORMATTED_ROWS + 1}")
        guide = wb.create_sheet("Hướng dẫn")
        for line in [
            "HƯỚNG DẪN ĐĂNG KÝ XE BẰNG EXCEL",
            "1. Sang sheet 'Xe đăng ký', mỗi xe một dòng, bắt đầu từ dòng 2.",
            "2. Bắt buộc: Biển số, Chủ xe. Biển số gõ kiểu 29A-123.45 hoặc 29A12345 đều được.",
            "3. Số điện thoại dạng 0901234567. Loại xe chọn từ danh sách.",
            "4. Nhấn Ctrl+S để lưu. Chương trình nhận diện tự đọc lại khi file thay đổi.",
            "5. Không xóa dòng tiêu đề (dòng 1). Không đổi tên sheet 'Xe đăng ký' / thứ tự sheet.",
        ]:
            guide.append([line])
        guide.column_dimensions["A"].width = 95
        guide["A1"].font = Font(bold=True, size=13)
        wb.save(self.path)

    # ---------- Đọc ----------
    def _stat(self):
        s = self.path.stat()
        return (s.st_mtime_ns, s.st_size)

    def _read(self):
        with self._lock:
            sig = self._stat()
            if self._cache is not None and sig == self._sig:
                return self._cache
            wb = load_workbook(self.path, read_only=True, data_only=True)
            ws = wb.worksheets[0]
            rows = []
            for rno, cells in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                vals = [_s(c) for c in (list(cells) + [None] * 8)[:8]]
                plate = normalize_plate(vals[0])
                if not plate:
                    continue
                d = dict(zip(KEYS, vals))
                d["plate"], d["id"] = plate, rno
                ph = d["phone"]
                if ph.isdigit() and not ph.startswith("0") and len(ph) in (9, 10):
                    d["phone"] = "0" + ph   # Excel hay làm mất số 0 đầu
                rows.append(d)
            wb.close()
            self._cache, self._sig = rows, sig
            return rows

    def _write(self, op):
        with self._lock:
            try:
                wb = load_workbook(self.path)
                result = op(wb.worksheets[0])
                wb.save(self.path)
            except PermissionError:
                raise ValueError(LOCKED_MSG)
            self._cache = None
            return result

    @staticmethod
    def _first_empty_row(ws):
        r = 2
        while _s(ws.cell(r, 1).value):
            r += 1
        return r

    # ---------- Đăng ký / sửa / xóa ----------
    def register(self, plate, owner, phone="", vehicle_type="", brand="", color="", note=""):
        plate = validate_vehicle(plate, owner, phone)
        if self.get(plate):
            raise ValueError(f"Biển số {plate} đã được đăng ký trước đó")

        def op(ws):
            r = self._first_empty_row(ws)
            for c, v in enumerate([pretty(plate), owner.strip(), phone.strip(), vehicle_type, brand, color, note, now()], 1):
                ws.cell(r, c).value = v
            return r
        return self._write(op)

    def update(self, plate, **fields):
        fields = {k: v for k, v in fields.items() if k in EDITABLE and v is not None}
        if not fields:
            raise ValueError("Không có trường nào để cập nhật")
        row = self.get(plate)
        if not row:
            raise ValueError(f"Không tìm thấy biển số {plate}")

        def op(ws):
            for k, v in fields.items():
                ws.cell(row["id"], KEYS.index(k) + 1).value = v
        self._write(op)

    def delete(self, plate):
        row = self.get(plate)
        if not row:
            raise ValueError(f"Không tìm thấy biển số {plate}")
        self._write(lambda ws: ws.delete_rows(row["id"]))

    # ---------- Tra cứu ----------
    def get(self, plate):
        plate = normalize_plate(plate)
        return next((dict(r) for r in self._read() if r["plate"] == plate), None)

    def find_vehicle(self, plate, max_dist=0):
        plate = normalize_plate(plate)
        row = self.get(plate)
        if row or max_dist == 0 or len(plate) < 7:
            return row
        cands = sorted(((edit_distance(plate, r["plate"]), r) for r in self._read()), key=lambda x: x[0])
        cands = [c for c in cands if c[0] <= max_dist]
        if len(cands) == 1 or (len(cands) > 1 and cands[0][0] < cands[1][0]):
            return dict(cands[0][1])
        return None

    def list_vehicles(self, keyword=None):
        rows = [dict(r) for r in self._read()]
        if keyword:
            k = keyword.lower()
            rows = [r for r in rows if any(k in str(r[f]).lower() for f in ("plate", "owner_name", "phone", "brand"))]
        return sorted(rows, key=lambda r: r["registered_at"], reverse=True)

    # ---------- Nhật ký (CSV) ----------
    def log_access(self, plate, vehicle_id=None, det_conf=None, ocr_conf=None, image_path=""):
        owner = ""
        if vehicle_id is not None:
            owner = next((r["owner_name"] for r in self._read() if r["id"] == vehicle_id), "")
        new = not self.log_path.exists()
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(LOG_FIELDS)
                w.writerow([now(), normalize_plate(plate), int(vehicle_id is not None), owner,
                            "" if det_conf is None else round(det_conf, 3),
                            "" if ocr_conf is None else round(ocr_conf, 3), image_path])
        except PermissionError:
            if not self._warned:   # đang mở file log bằng Excel -> bỏ qua, không làm sập chương trình
                print(f"Cảnh báo: không ghi được {self.log_path} (đang mở trong Excel?). Bỏ qua lượt này.")
                self._warned = True

    def list_logs(self, limit=50):
        if not self.log_path.exists():
            return []
        with open(self.log_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows[-limit:][::-1]:
            r["registered"] = int(r["registered"] or 0)
            out.append(r)
        return out

    def export_csv(self, out):
        rows = self.list_vehicles()
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(KEYS)
            w.writerows([[r[k] for k in KEYS] for r in rows])
        return len(rows)
