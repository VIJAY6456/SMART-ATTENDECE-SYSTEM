import face_recognition
import cv2

image_path = "dataset/2400031317_VIJAY/0.jpg"

# Check basic image properties
img_cv = cv2.imread(image_path)
print("Image shape (cv2):", img_cv.shape if img_cv is not None else "FAILED TO LOAD")

# Try face_recognition's own loader and detector
image = face_recognition.load_image_file(image_path)
print("Image shape (face_recognition):", image.shape)

locations = face_recognition.face_locations(image)
print("Face locations found:", locations)

# Also try the CNN-free HOG model explicitly
locations_hog = face_recognition.face_locations(image, model="hog")
print("HOG model locations:", locations_hog)