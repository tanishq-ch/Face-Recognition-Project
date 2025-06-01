# import cv2
#
#
# def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
#     # Converting image to gray-scale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # detecting features in gray-scale image, returns coordinates, width and height of features
#     features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
#     coords = []
#     # drawing rectangle around the feature and labeling it
#     for (x, y, w, h) in features:
#         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#         # Predicting the id of the user
#         id, _ = clf.predict(gray_img[y:y+h, x:x+w])
#         # Check for id of user and label the rectangle accordingly
#         if id==0:
#             cv2.putText(img, "Tanishq", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
#         coords = [x, y, w, h]
#
#     return coords
#
# # Method to recognize the person
# def recognize(img, clf, faceCascade):
#     color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
#     coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
#     return img
#
#
# # Loading classifier
# faceCascade = cv2.CascadeClassifier('C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Detection-Recognition-Using-OpenCV-in-Python-master/haarcascade_frontalface_default.xml')
#
# # Loading custom classifier to recognize
# clf = cv2.face.LBPHFaceRecognizer_create()
# clf.read("classifier_tanishq.yml")
#
# # Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Reading image from video stream
#     _, img = video_capture.read()
#     # Call method we defined above
#     img = recognize(img, clf, faceCascade)
#     # Writing processed image in a new window
#     cv2.imshow("face detection", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # releasing web-cam
# video_capture.release()
# # Destroying output window
# cv2.destroyAllWindows()
#
#
import cv2
import os
import numpy as np

def train_face_recognizer(dataset_path):
    faces = []
    labels = []
    label_dict = {}

    for root, dirs, files in os.walk(dataset_path):
        for idx, dir_name in enumerate(dirs):
            label_dict[idx] = dir_name
            person_path = os.path.join(root, dir_name)
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces.append(gray_img)
                labels.append(idx)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(labels))
    clf.save("classifier.yml")

    return clf, label_dict

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf, label_dict):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, confidence = clf.predict(gray_img[y:y+h, x:x+w])
        if confidence < 100:
            name = label_dict[id]
            cv2.putText(img, name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

def recognize(img, clf, faceCascade, label_dict):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    draw_boundary(img, faceCascade, 1.1, 10, color["white"], clf, label_dict)
    return img

# Set the path to your dataset
dataset_path = "C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Recognition-Project/Face-Detection-Recognition-Using-OpenCV-in-Python-master/dataset"

# Train the face recognizer
clf, label_dict = train_face_recognizer(dataset_path)

# Load classifier
faceCascade = cv2.CascadeClassifier('C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Recognition-Project/Face-Detection-Recognition-Using-OpenCV-in-Python-master/haarcascade_frontalface_default.xml')

# Capture real-time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = recognize(img, clf, faceCascade, label_dict)
    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release web-cam
video_capture.release()
cv2.destroyAllWindows()
