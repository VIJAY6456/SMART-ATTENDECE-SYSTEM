import face_recognition
import cv2
import pickle
import csv
import os
from datetime import datetime

# Load known face encodings
with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Set up attendance file for today
attendance_dir = "attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

today = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_dir, f"{today}.csv")

# Track who's already marked today (avoid duplicate entries)
marked_today = set()
if os.path.exists(attendance_file):
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                marked_today.add(row[0])
else:
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RollNo_Name", "Time"])

def mark_attendance(person_id):
    if person_id not in marked_today:
        marked_today.add(person_id)
        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([person_id, datetime.now().strftime("%H:%M:%S")])
        print(f"Attendance marked: {person_id}")

video_capture = cv2.VideoCapture(0)
print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)

        # Draw box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Attendance System - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print(f"Session ended. Attendance saved to {attendance_file}")