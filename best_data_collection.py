import cv2
import os
import time

# Common dataset directory
dataset_dir = "C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Recognition-Project/Face-Detection-Recognition-Using-OpenCV-in-Python-master/dataset"

# Create the dataset directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


# Method to generate a dataset to recognize a person
def generate_dataset(img, person_name, img_id):
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    img_path = os.path.join(person_folder, f"{person_name}_{img_id}.jpg")
    cv2.imwrite(img_path, img)


# Method to draw a boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting the image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecting features in gray-scale image, returns coordinates, width, and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # Drawing a rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, img_id, person_name):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If a feature is detected, the draw_boundary method will return the x, y coordinates and width and height of the rectangle else the length of coords will be 0
    if len(coords) == 4:
        # Updating the region of interest by cropping the image
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # Generate a dataset image and increment img_id
        generate_dataset(roi_img, person_name, img_id)
    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier(
    'C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Recognition-Project/Face-Detection-Recognition-Using-OpenCV-in-Python-master/haarcascade_frontalface_default.xml')

# Capturing real-time video stream (0 for built-in webcams, 0 or -1 for external webcams)
video_capture = cv2.VideoCapture(0)

# Initialize img_id with 0
img_id = 0

# Prompt the user to enter the person's name
person_name = input("Enter the person's name: ")

# Create a window to display the live video feed
cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)

while True:
    if img_id % 50 == 0:
        print("Collected", img_id, "images")

    # Reading an image from the video stream
    _, img = video_capture.read()

    # Call the method we defined above
    img = detect(img, faceCascade, img_id, person_name)

    # Writing processed image in the "face detection" window
    cv2.imshow("face detection", img)

    img_id += 1

    # Capture a specified number of images (e.g., 10) before stopping
    if img_id >= 100:
        break

    # Display the live video feed in the "Live Video Feed" window
    cv2.imshow("Live Video Feed", img)

    # Add a delay to slow down image capturing (1 frame per second)
    time.sleep(0.5)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam
video_capture.release()
# Destroying the output windows
cv2.destroyAllWindows()
