from flask import Flask, render_template, request
import csv
import os
from datetime import datetime
import face_recognition
import pickle
import base64
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Use the persistent disk path on Render if set, otherwise use local folders (for your PC)
DATA_ROOT = os.environ.get("DATA_ROOT", ".")
ATTENDANCE_DIR = os.path.join(DATA_ROOT, "attendance")
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
ENCODINGS_PATH = os.path.join(DATA_ROOT, "encodings", "encodings.pkl")

os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)

# Load known faces if the encodings file already exists, otherwise start empty
if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        known_data = pickle.load(f)
    known_encodings = known_data["encodings"]
    known_names = known_data["names"]
else:
    known_encodings = []
    known_names = []

def get_available_dates():
    if not os.path.exists(ATTENDANCE_DIR):
        return []
    files = [f.replace(".csv", "") for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")]
    return sorted(files, reverse=True)

def read_attendance(date):
    file_path = os.path.join(ATTENDANCE_DIR, f"{date}.csv")
    records = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    records.append({"id": row[0], "time": row[1]})
    return records

def mark_attendance_live(person_id):
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(ATTENDANCE_DIR, f"{today}.csv")
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(["RollNo_Name", "Time"])

    already_marked = set()
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                already_marked.add(row[0])

    if person_id not in already_marked:
        with open(file_path, "a", newline="") as f:
            csv.writer(f).writerow([person_id, datetime.now().strftime("%H:%M:%S")])
        return True
    return False

@app.route("/")
def index():
    dates = get_available_dates()
    selected_date = request.args.get("date", dates[0] if dates else None)
    records = read_attendance(selected_date) if selected_date else []
    return render_template("index.html", dates=dates, selected_date=selected_date, records=records)

@app.route("/mark")
def mark_page():
    return render_template("mark.html")

@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if not face_encodings:
        return {"status": "no_face"}

    face_encoding = face_encodings[0]

    if len(known_encodings) == 0:
        return {"status": "no_match"}

    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_names[best_match_index]
        newly_marked = mark_attendance_live(name)
        return {"status": "marked" if newly_marked else "already_marked", "name": name}

    return {"status": "unknown"}

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    name = data.get("name", "").strip()
    roll_no = data.get("roll_no", "").strip()
    image_data = data.get("image", "")

    if not name or not roll_no or not image_data:
        return {"status": "error", "message": "Missing name, roll number, or image"}

    person_id = f"{roll_no}_{name}"
    person_dir = os.path.join(DATASET_DIR, person_id)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    existing = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    count = len(existing)

    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return {"status": "no_face", "count": count}

    img_path = os.path.join(person_dir, f"{count}.jpg")
    Image.fromarray(frame).save(img_path)

    return {"status": "captured", "count": count + 1}

@app.route("/api/finalize_registration", methods=["POST"])
def finalize_registration():
    known_encodings_new = []
    known_names_new = []

    for person_folder in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(person_path):
            continue
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_encodings_new.append(encoding)
                known_names_new.append(person_folder)

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings_new, "names": known_names_new}, f)

    global known_encodings, known_names
    known_encodings = known_encodings_new
    known_names = known_names_new

    return {"status": "done", "total_people": len(set(known_names_new))}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))