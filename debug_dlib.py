import dlib
import cv2

print("dlib version:", dlib.__version__)

img = cv2.imread("dataset/2400031317_VIJAY/0.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = dlib.get_frontal_face_detector()
faces = detector(img_rgb, 1)

print("Number of faces found by dlib directly:", len(faces))
for f in faces:
    print(f)