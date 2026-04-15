from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import cv2
import numpy as np
import os
import json
from datetime import date, datetime, timedelta

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Database config ──
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, "instance", "intire.db")
os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ✅ THIS MUST COME BEFORE MODELS
db = SQLAlchemy(app)

# ────────────────────────────────────────────
#  MODELS (matches your schema exactly)
# ────────────────────────────────────────────

class Role(db.Model):
    __tablename__ = 'role_tbl'
    role_code = db.Column(db.Integer, primary_key=True)
    role_desc = db.Column(db.String(20))

class Account(db.Model):
    __tablename__ = 'account_tbl'
    account_no = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fname      = db.Column(db.String(50))
    mname      = db.Column(db.String(50))
    lname      = db.Column(db.String(50))
    mobile_no  = db.Column(db.String(20))
    email      = db.Column(db.String(100), unique=True)
    password   = db.Column(db.String(255))
    role_code  = db.Column(db.Integer, db.ForeignKey('role_tbl.role_code'), default=1)

class Inspection(db.Model):
    __tablename__ = 'inspection_tbl'
    inspection_no   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    plate_no        = db.Column(db.String(20))
    inspection_date = db.Column(db.String(20))
    vehicle_type    = db.Column(db.String(50))
    vehicle_model   = db.Column(db.String(100))
    inspector       = db.Column(db.String(100))
    date_inspected  = db.Column(db.String(20))

class DefectType(db.Model):
    __tablename__ = 'defect_tbl'
    defect_code = db.Column(db.Integer, primary_key=True)
    defect_type = db.Column(db.String(50))

# Add image_data column to TireDefect model
class TireDefect(db.Model):
    __tablename__ = 'tiredefects_tbl'
    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    plate_no         = db.Column(db.String(20), db.ForeignKey('inspection_tbl.plate_no'))
    tire_position    = db.Column(db.String(30))
    tire_dot         = db.Column(db.String(4))
    manufacture_date = db.Column(db.String(50))
    expiry_date      = db.Column(db.String(50))
    tire_age         = db.Column(db.String(20))
    validity         = db.Column(db.String(20))
    defect_code      = db.Column(db.Integer, db.ForeignKey('defect_tbl.defect_code'))
    image_data       = db.Column(db.Text)  # ← NEW: stores base64 image

class Notification(db.Model):
    __tablename__ = 'notification_tbl'
    notification_no      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    notification_content = db.Column(db.Text)
    notification_type    = db.Column(db.Integer, db.ForeignKey('notification_type_tbl.notification_type'))

class NotificationType(db.Model):
    __tablename__ = 'notification_type_tbl'
    notification_type = db.Column(db.Integer, primary_key=True)
    notification_desc = db.Column(db.String(20))


# ── Seed default data & create tables ──
with app.app_context():
    db.create_all()

    # Seed roles if empty
    if not Role.query.first():
        db.session.add_all([
            Role(role_code=1, role_desc='user'),
            Role(role_code=2, role_desc='admin'),
            Role(role_code=3, role_desc='guest'),
        ])

    # Seed defect types if empty
    if not DefectType.query.first():
        db.session.add_all([
            DefectType(defect_code=1, defect_type='Surface Crack'),
            DefectType(defect_code=2, defect_type='Bulge'),
            DefectType(defect_code=3, defect_type='Worn Tread'),
            DefectType(defect_code=4, defect_type='Puncture Hole'),
            DefectType(defect_code=5, defect_type='Puncture Object'),
        ])

    # Seed notification types if empty
    if not NotificationType.query.first():
        db.session.add_all([
            NotificationType(notification_type=1, notification_desc='alerts'),
            NotificationType(notification_type=2, notification_desc='info'),
            NotificationType(notification_type=3, notification_desc='system'),
        ])

    db.session.commit()


# ── Load YOLO model ──
model = YOLO("assets\\ai model\\best.pt")


# ────────────────────────────────────────────
#  SERVE HTML
# ────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "overview.html")

@app.route("/<path:path>")
def serve_file(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    return "File not found", 404


# ────────────────────────────────────────────
#  PREDICT
# ────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img  = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)

    detections = []
    for r in results:
        for b in r.boxes:
            detections.append({
                "class":      int(b.cls[0]),
                "confidence": float(b.conf[0]),
                "bbox":       b.xyxy[0].tolist()
            })

    return jsonify(detections)


# ────────────────────────────────────────────
#  INSPECTION ROUTES
# ────────────────────────────────────────────

# Replace the save_inspection route
@app.route("/save-inspection", methods=["POST"])
def save_inspection():
    data = request.get_json()
    inspection_no = data.get('inspectionNo')  # None on first call, ID on "Add Another"

    # 1. Only create a new Inspection record if this is the first save
    if not inspection_no:
        inspection = Inspection(
            plate_no        = data.get('plateNumber'),
            inspection_date = data.get('date'),
            vehicle_type    = data.get('carType'),
            vehicle_model   = data.get('vehicleModel'),
            inspector       = data.get('inspectedBy'),
            date_inspected  = data.get('date'),
        )
        db.session.add(inspection)
        db.session.flush()
        inspection_no = inspection.inspection_no

        # Save notification only on first inspection save
        notif = Notification(
            notification_content = f"Inspection completed for plate {data.get('plateNumber')}",
            notification_type    = 2
        )
        db.session.add(notif)
    else:
        # Fetch existing inspection to get plate_no
        inspection = Inspection.query.get(inspection_no)
        if not inspection:
            return jsonify({'success': False, 'error': 'Inspection not found'}), 404

    # 2. Save tire defects with the captured image
    defects = data.get('defects', [])
    image_data = data.get('imageBase64', '')

    for defect_code in defects:
        tire = TireDefect(
            plate_no         = data.get('plateNumber') or inspection.plate_no,
            tire_position    = data.get('tirePosition'),
            tire_dot         = data.get('dotCode'),
            manufacture_date = data.get('manufactureDate'),
            expiry_date      = data.get('expiryDate'),
            tire_age         = data.get('tireAge'),
            validity         = data.get('validity'),
            defect_code      = defect_code,
            image_data       = image_data  # ← photo saved per defect entry
        )
        db.session.add(tire)

    db.session.commit()
    return jsonify({ 'success': True, 'inspection_no': inspection_no })

@app.route("/inspection-history", methods=["GET"])
def get_inspection_history():
    inspections = Inspection.query.order_by(Inspection.inspection_no.desc()).all()
    result = []
    for insp in inspections:
        tires = TireDefect.query.filter_by(plate_no=insp.plate_no).all()
        defect_list = []
        for t in tires:
            defect = DefectType.query.get(t.defect_code)
            defect_list.append({
                'tirePosition':    t.tire_position,
                'dotCode':         t.tire_dot,
                'manufactureDate': t.manufacture_date,
                'expiryDate':      t.expiry_date,
                'tireAge':         t.tire_age,
                'validity':        t.validity,
                'defectType':      defect.defect_type if defect else 'Unknown',
                'imageData':       t.image_data or ''
            })
        result.append({
            'inspectionNo':  insp.inspection_no,
            'plateNo':       insp.plate_no,
            'vehicleType':   insp.vehicle_type,
            'vehicleModel':  insp.vehicle_model,
            'inspector':     insp.inspector,
            'dateInspected': insp.date_inspected,
            'tires':         defect_list
        })
    return jsonify(result)

# ────────────────────────────────────────────
#  DEFECT TYPES
# ────────────────────────────────────────────

@app.route("/defects", methods=["GET"])
def get_defects():
    defects = DefectType.query.all()
    return jsonify([{
        'defectCode': d.defect_code,
        'defectType': d.defect_type
    } for d in defects])


# ────────────────────────────────────────────
#  NOTIFICATIONS
# ────────────────────────────────────────────

@app.route("/notifications", methods=["GET"])
def get_notifications():
    notifs = Notification.query.order_by(Notification.notification_no.desc()).all()
    return jsonify([{
        'notificationNo':      n.notification_no,
        'notificationContent': n.notification_content,
        'notificationType':    n.notification_type,
    } for n in notifs])

def _parse_yyyy_mm_dd(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None

@app.route("/analytics", methods=["GET"])
def analytics():
    """
    Analytics aggregates for analytics.html.
    Query param: period = week|month|year (default: week)
    """
    period = (request.args.get("period") or "week").lower()
    today = date.today()

    # Define buckets
    if period == "year":
        start = date(today.year, 1, 1)
        end = today
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        bars = [0] * 12

        def bucket_index(d: date) -> int:
            return d.month - 1

    elif period == "month":
        # Show 4 week buckets (rolling 28 days)
        end = today
        start = today - timedelta(days=27)
        labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
        bars = [0] * 4

        def bucket_index(d: date) -> int:
            # 0..3 across the 28-day window
            delta = (d - start).days
            idx = int(delta // 7)
            return max(0, min(3, idx))

    else:
        # week (default): last 7 days incl today
        end = today
        start = today - timedelta(days=6)
        # Labels Mon..Sun for the current 7-day window ending today
        labels = []
        bars = [0] * 7
        for i in range(7):
            d = start + timedelta(days=i)
            labels.append(d.strftime("%a"))

        def bucket_index(d: date) -> int:
            return (d - start).days

    # Load inspections in range (best-effort parse because dates are stored as strings)
    inspections = Inspection.query.all()
    inspections_in_range = []
    for insp in inspections:
        d = _parse_yyyy_mm_dd(insp.date_inspected or insp.inspection_date)
        if not d:
            continue
        if start <= d <= end:
            inspections_in_range.append((insp, d))

    # Bars: inspections over time
    for _, d in inspections_in_range:
        idx = bucket_index(d)
        if 0 <= idx < len(bars):
            bars[idx] += 1

    # Totals
    total_inspections = len(inspections_in_range)

    # Active/resolved definition:
    # - active: inspection has >=1 tiredefects_tbl rows
    # - resolved: inspection has 0 tiredefects_tbl rows
    plate_nos = [insp.plate_no for insp, _ in inspections_in_range if insp.plate_no]
    active_plate_nos = set()
    if plate_nos:
        active_plate_nos = set(
            r[0] for r in db.session.query(TireDefect.plate_no)
            .filter(TireDefect.plate_no.in_(plate_nos))
            .distinct()
            .all()
        )

    active_issues = sum(1 for insp, _ in inspections_in_range if insp.plate_no in active_plate_nos)
    resolved = max(0, total_inspections - active_issues)
    rate = f"{int(round((resolved / total_inspections) * 100))}%" if total_inspections else "0%"

    # Issue breakdown (counts of defect types in range)
    inspection_plates_in_range = set(plate_nos)
    defect_counts = []
    if inspection_plates_in_range:
        rows = (
            db.session.query(DefectType.defect_type, func.count(TireDefect.id))
            .join(TireDefect, TireDefect.defect_code == DefectType.defect_code)
            .filter(TireDefect.plate_no.in_(inspection_plates_in_range))
            .group_by(DefectType.defect_type)
            .order_by(func.count(TireDefect.id).desc())
            .all()
        )
        defect_counts = [{"name": r[0], "count": int(r[1])} for r in rows]

    # "No Damage" = inspections with no defect rows
    no_damage_count = resolved
    defect_counts.append({"name": "No Damage", "count": int(no_damage_count)})

    # Most inspected vehicles (top 4)
    vehicle_counts = {}
    for insp, _ in inspections_in_range:
        plate = (insp.plate_no or "").strip()
        model = (insp.vehicle_model or "").strip()
        title = " · ".join([p for p in [model, plate] if p]) or f"Inspection #{insp.inspection_no}"
        vehicle_counts[title] = vehicle_counts.get(title, 0) + 1

    top_vehicles = sorted(vehicle_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
    vehicles = [{"name": n, "count": c} for n, c in top_vehicles]

    label_map = {"week": "This Week", "month": "This Month", "year": "This Year"}
    return jsonify({
        "period": period,
        "label": label_map.get(period, "This Week"),
        "bars": bars,
        "barLabels": labels,
        "stats": {
            "total": total_inspections,
            "active": active_issues,
            "resolved": resolved,
            "rate": rate
        },
        "breakdown": defect_counts,
        "vehicles": vehicles
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)