"""Giao diện web quản lý đăng ký biển số xe.

Chạy:  streamlit run app.py
"""
import re
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from db import DB_PATH, normalize_plate, open_db, pretty, validate_vehicle

VEHICLE_TYPES = ["Xe máy", "Ô tô", "Xe tải", "Xe đạp điện", "Khác"]

st.set_page_config(page_title="Quản lý biển số xe", page_icon="🚗", layout="wide")
st.title("🚗 Quản lý đăng ký biển số xe")


# ---------------- Kết nối CSDL ----------------
@st.cache_resource(show_spinner=False)
def get_db(backend, sqlite_path, mongo_uri, mongo_name, excel_path):
    return open_db(SimpleNamespace(backend=backend, db=sqlite_path, mongo_uri=mongo_uri,
                                   mongo_db=mongo_name, excel_file=excel_path))


with st.sidebar:
    st.header("⚙️ Cài đặt")
    choice = st.radio("Cơ sở dữ liệu", ["SQLite", "MongoDB", "Excel"], key="backend")
    excel_path = "vehicles.xlsx"
    if choice == "SQLite":
        sqlite_path = st.text_input("File .db", DB_PATH, key="db_path")
        mongo_uri, mongo_name = "", ""
    elif choice == "Excel":
        sqlite_path, mongo_uri, mongo_name = DB_PATH, "", ""
        excel_path = st.text_input("File Excel", "vehicles.xlsx", key="excel_path")
    else:
        sqlite_path = DB_PATH
        mongo_uri = st.text_input("Chuỗi kết nối", "mongodb://localhost:27017", key="mongo_uri")
        mongo_name = st.text_input("Tên database", "lpr", key="mongo_db")
    weights = st.text_input("Model YOLO (tuỳ chọn, để điền biển số từ ảnh)",
                            "runs/detect/plate_det/weights/best.pt", key="weights")

try:
    db = get_db({"SQLite": "sqlite", "MongoDB": "mongo", "Excel": "excel"}[choice],
                sqlite_path, mongo_uri, mongo_name, excel_path)
except (SystemExit, Exception) as e:  # MongoPlateDB báo lỗi kết nối bằng SystemExit
    st.error(f"Không kết nối được CSDL: {e}")
    st.stop()


def rd(row):
    """Row/dict -> dict thường, bỏ khóa id nội bộ."""
    d = dict(row)
    d.pop("_id", None)
    d.pop("id", None)
    return d


def flash(msg, kind="success"):
    st.session_state["flash"] = (kind, msg)
    st.rerun()


if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    getattr(st, kind)(msg)

if "ver" not in st.session_state:   # đổi số phiên bản -> form được làm trống sau khi đăng ký thành công
    st.session_state["ver"] = 0
ver = st.session_state["ver"]


def k(name):
    return f"{name}_{ver}"


# ---------------- Nhận diện biển số từ ảnh (tuỳ chọn) ----------------
@st.cache_resource(show_spinner="Đang tải model...")
def load_models(weights_path):
    import easyocr
    from ultralytics import YOLO
    return YOLO(weights_path), easyocr.Reader(["en"], gpu=False)


def on_detect():
    up = st.session_state.get("upload")
    st.session_state.pop("detect_crop", None)
    if up is None:
        st.session_state["detect_msg"] = ("warning", "Hãy chọn một ảnh trước.")
        return
    try:
        import cv2
        import numpy as np

        import lpr
        detector, reader = load_models(st.session_state["weights"])
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        res = detector.predict(img, conf=0.4, verbose=False)[0]
        if len(res.boxes) == 0:
            st.session_state["detect_msg"] = ("warning", "Không tìm thấy biển số trong ảnh.")
            return
        box = max(res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4]
        text, conf = lpr.read_plate(reader, crop)
        st.session_state["detect_crop"] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        st.session_state[k("plate")] = text
        st.session_state["detect_msg"] = (
            "success", f"Đọc được: {pretty(text)} (độ tin cậy {conf:.2f}). Kiểm tra lại rồi sửa nếu sai.")
    except Exception as e:
        st.session_state["detect_msg"] = ("error", f"Không nhận diện được: {e}")


tab_add, tab_list, tab_logs = st.tabs(["➕ Đăng ký xe", "📋 Danh sách xe", "🕒 Nhật ký nhận diện"])

# ================= Tab 1: Form đăng ký =================
with tab_add:
    with st.expander("📷 Điền biển số từ ảnh (tuỳ chọn)"):
        st.file_uploader("Ảnh xe / biển số", type=["jpg", "jpeg", "png"], key="upload")
        st.button("Nhận diện biển số", on_click=on_detect)
        if "detect_msg" in st.session_state:
            kind, msg = st.session_state["detect_msg"]
            getattr(st, kind)(msg)
        if "detect_crop" in st.session_state:
            st.image(st.session_state["detect_crop"], width=260, caption="Biển số được cắt")

    with st.form("register_form"):
        c1, c2 = st.columns(2)
        plate = c1.text_input("Biển số *", key=k("plate"), placeholder="29A-123.45")
        owner = c2.text_input("Họ tên chủ xe *", key=k("owner"))
        phone = c1.text_input("Số điện thoại", key=k("phone"), placeholder="0901234567")
        vtype = c2.selectbox("Loại xe", VEHICLE_TYPES, key=k("vtype"))
        brand = c1.text_input("Hãng xe", key=k("brand"), placeholder="Honda, Toyota...")
        color = c2.text_input("Màu xe", key=k("color"))
        note = st.text_area("Ghi chú", key=k("note"))
        submitted = st.form_submit_button("Đăng ký", type="primary")

    if submitted:
        try:
            db.register(plate, owner, phone, vtype, brand.strip(), color.strip(), note.strip())
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state["ver"] += 1
            for key in ("detect_msg", "detect_crop"):
                st.session_state.pop(key, None)
            flash(f"✅ Đã đăng ký {pretty(normalize_plate(plate))} – {owner.strip()}")

# ================= Tab 2: Danh sách / sửa / xóa =================
with tab_list:
    kw = st.text_input("🔍 Tìm kiếm (biển số, tên, SĐT, hãng)", key="kw").strip()
    rows = [rd(r) for r in db.list_vehicles(kw or None)]
    if not rows and kw and normalize_plate(kw):   # gõ "29A-123.45" vẫn tìm được
        rows = [rd(r) for r in db.list_vehicles(normalize_plate(kw))]

    st.metric("Số xe", len(rows))
    if not rows:
        st.info("Chưa có xe nào." if not kw else "Không tìm thấy kết quả.")
    else:
        df = pd.DataFrame(rows)
        df["plate"] = df["plate"].map(pretty)
        df = df.rename(columns={"plate": "Biển số", "owner_name": "Chủ xe", "phone": "SĐT",
                                "vehicle_type": "Loại xe", "brand": "Hãng", "color": "Màu",
                                "note": "Ghi chú", "registered_at": "Ngày đăng ký"})
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "vehicles.csv", "text/csv")

        st.divider()
        st.subheader("Sửa / xóa")
        by_plate = {r["plate"]: r for r in rows}
        sel = st.selectbox("Chọn xe", list(by_plate),
                           format_func=lambda p: f"{pretty(p)} – {by_plate[p]['owner_name']}", key="sel")
        cur = by_plate[sel]
        types = VEHICLE_TYPES if cur["vehicle_type"] in VEHICLE_TYPES else VEHICLE_TYPES + [cur["vehicle_type"]]
        with st.form(f"edit_{sel}"):
            e1, e2 = st.columns(2)
            e_owner = e1.text_input("Họ tên chủ xe", cur["owner_name"])
            e_phone = e2.text_input("Số điện thoại", cur["phone"])
            e_type = e1.selectbox("Loại xe", types, index=types.index(cur["vehicle_type"]) if cur["vehicle_type"] in types else 0)
            e_brand = e2.text_input("Hãng xe", cur["brand"])
            e_color = e1.text_input("Màu xe", cur["color"])
            e_note = e2.text_input("Ghi chú", cur["note"])
            save = st.form_submit_button("💾 Cập nhật")
        if save:
            try:
                validate_vehicle(sel, e_owner, e_phone)
                db.update(sel, owner_name=e_owner.strip(), phone=e_phone.strip(), vehicle_type=e_type,
                          brand=e_brand.strip(), color=e_color.strip(), note=e_note.strip())
            except ValueError as e:
                st.error(str(e))
            else:
                flash(f"✅ Đã cập nhật {pretty(sel)}")
        confirm = st.checkbox(f"Xác nhận xóa {pretty(sel)}", key=f"confirm_{sel}")
        if st.button("🗑️ Xóa xe này", disabled=not confirm):
            db.delete(sel)
            flash(f"🗑️ Đã xóa {pretty(sel)}", "info")

# ================= Tab 3: Nhật ký =================
with tab_logs:
    limit = st.slider("Số dòng hiển thị", 10, 500, 50, key="limit")
    logs = [dict(r) for r in db.list_logs(limit)]
    if not logs:
        st.info("Chưa có lượt nhận diện nào. Chạy lpr.py với camera để ghi nhật ký.")
    else:
        ldf = pd.DataFrame({
            "Thời gian": [r["seen_at"] for r in logs],
            "Biển số": [pretty(r["plate"]) for r in logs],
            "Trạng thái": ["✅ Đã đăng ký" if r["registered"] else "❌ Chưa đăng ký" for r in logs],
            "Chủ xe": [r.get("owner_name") or "" for r in logs],
        })
        a, b = st.columns(2)
        a.metric("Lượt đã đăng ký", sum(1 for r in logs if r["registered"]))
        b.metric("Lượt chưa đăng ký", sum(1 for r in logs if not r["registered"]))
        st.dataframe(ldf, hide_index=True)
