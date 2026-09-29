import face_recognition
import cv2
import pickle
import csv
import os
from datetime import datetime
from scipy.spatial import distance as dist

# Eye landmark indices (from dlib's 68-point model, used by face_recognition's face_landmarks)
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.21   # below this = eye considered closed
CONSEC_FRAMES = 2      # how many consecutive closed frames count as a blink

# Load known face encodings
with open("encodings/encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

attendance_dir = "attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

today = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_dir, f"{today}.csv")

marked_today = set()
if os.path.exists(attendance_file):
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
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
        print(f"Attendance marked (liveness verified): {person_id}")

# Track blink state per detected person across frames
blink_counters = {}
verified_people = set()

video_capture = cv2.VideoCapture(0)
print("Starting recognition with liveness check. Blink naturally to verify. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        status = "Unknown"
        color = (0, 0, 255)

        if name != "Unknown":
            if name in verified_people:
                status = "Verified"
                color = (0, 255, 0)
            else:
                # Calculate EAR for both eyes
                left_eye = landmarks.get("left_eye")
                right_eye = landmarks.get("right_eye")

                if left_eye and right_eye:
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    if name not in blink_counters:
                        blink_counters[name] = 0

                    if ear < EAR_THRESHOLD:
                        blink_counters[name] += 1
                    else:
                        if blink_counters[name] >= CONSEC_FRAMES:
                            verified_people.add(name)
                            mark_attendance(name)
                            status = "Verified"
                            color = (0, 255, 0)
                        blink_counters[name] = 0

                    if status != "Verified":
                        status = "Blink to verify..."
                        color = (0, 165, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} - {status}" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Attendance with Liveness Check - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print(f"Session ended. Attendance saved to {attendance_file}")