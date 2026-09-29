import face_recognition
import os
import pickle

dataset_dir = "dataset"
encodings_dir = "encodings"

if not os.path.exists(encodings_dir):
    os.makedirs(encodings_dir)

known_encodings = []
known_names = []

for person_folder in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_folder)
    if not os.path.isdir(person_path):
        continue

    print(f"Processing {person_folder}...")

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) > 0:
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_encodings.append(encoding)
            known_names.append(person_folder)  # format: "rollno_name"
        else:
            print(f"  No face found in {image_name}, skipping")

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}
with open(os.path.join(encodings_dir, "encodings.pkl"), "wb") as f:
    pickle.dump(data, f)

print(f"Done! Encoded {len(known_encodings)} face images for {len(set(known_names))} people.")