import json
import os
import cv2

def load_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            annotations.append(json.loads(line.strip()))
    return annotations

def preprocess_data(image_dir, annotation_file):
    annotations = load_annotations(annotation_file)
    data = []
    for anno in annotations:
        img_path = os.path.join(image_dir, anno['ID'] + '.jpg')
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        bboxes = anno['gtboxes']
        data.append((img, bboxes))
    return data

# Example usage
train_data = preprocess_data(
    image_dir="D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/CrowdHuman/CrowdHuman_train01",
    annotation_file="D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/CrowdHuman/annotation_train.odgt"
)



