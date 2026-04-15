from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import os
import json

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Database config ──
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///intire.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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

@app.route("/save-inspection", methods=["POST"])
def save_inspection():
    data = request.get_json()

    # 1. Save inspection record
    inspection = Inspection(
        plate_no        = data.get('plateNumber'),
        inspection_date = data.get('date'),
        vehicle_type    = data.get('carType'),
        vehicle_model   = data.get('vehicleModel'),
        inspector       = data.get('inspectedBy'),
        date_inspected  = data.get('date'),
    )
    db.session.add(inspection)
    db.session.flush()  # get inspection_no before commit

    # 2. Save tire defects
    defects = data.get('defects', [])
    for defect_code in defects:
        tire = TireDefect(
            plate_no         = data.get('plateNumber'),
            tire_position    = data.get('tirePosition'),
            tire_dot         = data.get('dotCode'),
            manufacture_date = data.get('manufactureDate'),
            expiry_date      = data.get('expiryDate'),
            tire_age         = data.get('tireAge'),
            validity         = data.get('validity'),
            defect_code      = defect_code
        )
        db.session.add(tire)

    # 3. Save notification
    notif = Notification(
        notification_content = f"Inspection completed for plate {data.get('plateNumber')}",
        notification_type    = 2  # info
    )
    db.session.add(notif)

    db.session.commit()
    return jsonify({ 'success': True, 'inspection_no': inspection.inspection_no })


@app.route("/inspections", methods=["GET"])
def get_inspections():
    records = Inspection.query.order_by(Inspection.inspection_no.desc()).all()
    return jsonify([{
        'inspectionNo':  r.inspection_no,
        'plateNo':       r.plate_no,
        'vehicleType':   r.vehicle_type,
        'vehicleModel':  r.vehicle_model,
        'inspector':     r.inspector,
        'dateInspected': r.date_inspected,
    } for r in records])


@app.route("/inspections/<int:id>", methods=["GET"])
def get_inspection(id):
    r       = Inspection.query.get_or_404(id)
    defects = TireDefect.query.filter_by(plate_no=r.plate_no).all()
    return jsonify({
        'inspectionNo':  r.inspection_no,
        'plateNo':       r.plate_no,
        'vehicleType':   r.vehicle_type,
        'vehicleModel':  r.vehicle_model,
        'inspector':     r.inspector,
        'dateInspected': r.date_inspected,
        'defects': [{
            'tirePosition':    d.tire_position,
            'tireDot':         d.tire_dot,
            'manufactureDate': d.manufacture_date,
            'expiryDate':      d.expiry_date,
            'tireAge':         d.tire_age,
            'validity':        d.validity,
            'defectCode':      d.defect_code,
        } for d in defects]
    })


@app.route("/inspections/<int:id>", methods=["DELETE"])
def delete_inspection(id):
    record = Inspection.query.get_or_404(id)
    db.session.delete(record)
    db.session.commit()
    return jsonify({ 'success': True })


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)