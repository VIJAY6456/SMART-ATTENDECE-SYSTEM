import cv2
import os

# Ask for student details
name = input("Enter student name: ")
roll_no = input("Enter roll number: ")

# Create dataset folder if it doesn't exist
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Create a subfolder for this student
person_dir = os.path.join(dataset_dir, f"{roll_no}_{name}")
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
max_images = 10

print("Look at the camera. Capturing images... Press 'q' to stop early.")

while count < max_images:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_path = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)  # save the FULL frame, not the crop
        count += 1
        print(f"Captured image {count}/{max_images}")

    cv2.imshow('Registering Face - Press q to stop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print(f"Registration complete. {count} images saved for {name}.")