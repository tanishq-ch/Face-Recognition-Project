# import numpy as np
# from PIL import Image
# import os, cv2
#
# # Method to train custom classifier to recognize face
# def train_classifer(data_dir):
#     # Read all the images in custom data-set
#     path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
#     faces = []
#     ids = []
#
#     # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
#     for image in path:
#         img = Image.open(image).convert('L')
#         imageNp = np.array(img, 'uint8')
#         try:
#             id = int(os.path.split(image)[1].split("_")[1].split(".")[0])
#         except ValueError:
#             # Handle thex case where the filename format doesn't match the expected pattern
#             # You can assign a default value or decide on a specific behavior in this case
#             id = 0  # or any other default value
#
#         faces.append(imageNp)
#         ids.append(id)
#
#     ids = np.array(ids)
#
#     # Train and save classifier
#     clf = cv2.face.LBPHFaceRecognizer_create()
#     clf.train(faces, ids)
#     clf.write("classifier.yml")
#
#
# train_classifer("dataset/")

import numpy as np
from PIL import Image
import os, cv2


# Method to train custom classifier to recognize face
def train_classifier(data_dir):
    for user_folder in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_folder)

        if os.path.isdir(user_path):
            faces = []
            ids = []

            for image_file in os.listdir(user_path):
                image_path = os.path.join(user_path, image_file)
                img = Image.open(image_path).convert('L')
                imageNp = np.array(img, 'uint8')

                try:
                    id = int(image_file.split("_")[1].split(".")[0])
                except ValueError:
                    id = -1  # or any other default value

                faces.append(imageNp)
                ids.append(id)

            ids = np.array(ids)

            # Train and save classifier for each user
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces, ids)

            # Save the classifier with a unique name for each user
            classifier_filename = f"classifier_{user_folder}.yml"
            clf.write(classifier_filename)
            print(f"Classifier trained and saved for user {user_folder}.")


# Train classifier for all users in the "dataset" folder
train_classifier("C:/Users/toshu/PycharmProjects/pblprojectfacerecognition/Face-Recognition-Project/Face-Detection-Recognition-Using-OpenCV-in-Python-master/dataset/")
