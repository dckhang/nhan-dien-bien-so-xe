"""Backend MongoDB cho hệ thống đăng ký biển số (cùng giao diện với PlateDB trong db.py).

Collections:
  vehicles    { _id, plate (unique), owner_name, phone, vehicle_type, brand, color, note, registered_at }
  access_logs { _id, plate, vehicle_id, owner_name, registered, det_conf, ocr_conf, image_path, seen_at }
"""
import csv

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from db import EDITABLE, edit_distance, normalize_plate, now, validate_vehicle

FIELDS = ["plate", "owner_name", "phone", "vehicle_type", "brand", "color", "note", "registered_at"]


def _with_id(doc):
    """Thêm khóa 'id' để code dùng chung với bản SQLite (vehicle['id'])."""
    if doc is not None:
        doc["id"] = doc["_id"]
    return doc


class MongoPlateDB:
    def __init__(self, uri="mongodb://localhost:27017", db_name="lpr", client=None):
        if client is None:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            try:
                client.admin.command("ping")
            except Exception as e:
                raise SystemExit(f"Không kết nối được MongoDB tại {uri}: {e}")
        self.client = client
        self.db = client[db_name]
        self.vehicles = self.db["vehicles"]
        self.logs = self.db["access_logs"]
        self.vehicles.create_index([("plate", ASCENDING)], unique=True)
        self.logs.create_index([("plate", ASCENDING)])
        self.logs.create_index([("seen_at", DESCENDING)])

    # ---------- Đăng ký / sửa / xóa ----------
    def register(self, plate, owner, phone="", vehicle_type="", brand="", color="", note=""):
        plate = validate_vehicle(plate, owner, phone)
        doc = {"plate": plate, "owner_name": owner.strip(), "phone": phone.strip(),
               "vehicle_type": vehicle_type, "brand": brand, "color": color,
               "note": note, "registered_at": now()}
        try:
            return self.vehicles.insert_one(doc).inserted_id
        except DuplicateKeyError:
            raise ValueError(f"Biển số {plate} đã được đăng ký trước đó")

    def update(self, plate, **fields):
        fields = {k: v for k, v in fields.items() if k in EDITABLE and v is not None}
        if not fields:
            raise ValueError("Không có trường nào để cập nhật")
        res = self.vehicles.update_one({"plate": normalize_plate(plate)}, {"$set": fields})
        if res.matched_count == 0:
            raise ValueError(f"Không tìm thấy biển số {plate}")

    def delete(self, plate):
        res = self.vehicles.delete_one({"plate": normalize_plate(plate)})
        if res.deleted_count == 0:
            raise ValueError(f"Không tìm thấy biển số {plate}")

    # ---------- Tra cứu ----------
    def get(self, plate):
        return _with_id(self.vehicles.find_one({"plate": normalize_plate(plate)}))

    def find_vehicle(self, plate, max_dist=0):
        plate = normalize_plate(plate)
        row = self.get(plate)
        if row or max_dist == 0 or len(plate) < 7:
            return row
        cands = []
        for r in self.vehicles.find():
            d = edit_distance(plate, r["plate"])
            if d <= max_dist:
                cands.append((d, r))
        cands.sort(key=lambda x: x[0])
        if len(cands) == 1 or (len(cands) > 1 and cands[0][0] < cands[1][0]):
            return _with_id(cands[0][1])
        return None

    def list_vehicles(self, keyword=None):
        query = {}
        if keyword:
            rx = {"$regex": keyword, "$options": "i"}
            query = {"$or": [{"plate": rx}, {"owner_name": rx}, {"phone": rx}, {"brand": rx}]}
        return [_with_id(d) for d in self.vehicles.find(query).sort("registered_at", DESCENDING)]

    # ---------- Nhật ký ----------
    def log_access(self, plate, vehicle_id=None, det_conf=None, ocr_conf=None, image_path=""):
        owner = None
        if vehicle_id is not None:
            v = self.vehicles.find_one({"_id": vehicle_id})
            owner = v["owner_name"] if v else None
        self.logs.insert_one({
            "plate": normalize_plate(plate), "vehicle_id": vehicle_id, "owner_name": owner,
            "registered": int(vehicle_id is not None), "det_conf": det_conf, "ocr_conf": ocr_conf,
            "image_path": image_path, "seen_at": now()})

    def list_logs(self, limit=50):
        return list(self.logs.find().sort("_id", DESCENDING).limit(limit))

    def export_csv(self, out):
        rows = self.list_vehicles()
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows([[r.get(k, "") for k in FIELDS] for r in rows])
        return len(rows)
