import os
import cv2
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 224
DATASET_PATH = "./oxford5k_data/images"
GROUND_TRUTH_PATH = "./oxford5k_data/groundtruth.json"

def process_images(dataset_path, groundtruth_path):
    images = []
    labels = []

    #Loading groundtruth data
    with open(groundtruth_path, 'r') as gt_file:
        groundtruth = json.load(gt_file)

    #Checking if groundtruth has content
    if not groundtruth:
        print("Groundtruth file is empty or not properly formatted.")
        return images, labels

    # Collected all labeled image names into set for lookup
    labeled_images = set()
    for category, data in groundtruth.items():
        for quality in ["ok", "good", "junk", "query"]:
            labeled_images.update(data[quality])

    # Processing only labeled images
    for img_name in os.listdir(dataset_path):
        if img_name in labeled_images:
            img_path = os.path.join(dataset_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resizing to fixed size
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize the image
                images.append(img)

                # Geting label by searching in groundtruth categories
                for category, data in groundtruth.items():
                    for quality in ["ok", "good", "junk", "query"]:
                        if img_name in data[quality]:
                            labels.append(category)
                            break
                    else:
                        continue
                    break

    #Converting lists to numpy arrays
    images = np.array(images)

    #what if no labels found?
    if not labels:
        print("No labels were found. Check if the groundtruth keys match the image filenames.")
        return images, labels

    labels = np.array(labels)

    #Encodeing labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)

    return images, labels, le

#Calling function to process data
images, labels, label_encoder = process_images(DATASET_PATH, GROUND_TRUTH_PATH)

#Saveing processed data for training
if len(images) > 0 and len(labels) > 0:
    np.save('images.npy', images)
    np.save('labels.npy', labels)
else:
    print("No valid images or labels were found. Please check dataset and ground truth.")
