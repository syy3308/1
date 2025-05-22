import os
import cv2
import numpy as np
import csv
import xml.etree.ElementTree as ET
import dlib

# VOC 数据集路径
voc_root = 'D:\\ProgramData\\VOCdevkit\\VOC2007'
person_image_dir = os.path.join(voc_root, 'JPEGImages')
person_annotation_dir = os.path.join(voc_root, 'Annotations')

# 初始化 OpenCV 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 初始化 Dlib 特征提取器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def return_128d_features(path_img):
    try:
        # 使用 OpenCV 读取图像
        img = cv2.imread(path_img)
        if img is None:
            print(f"无法读取图像: {path_img}")
            return None

        # 转换为灰度图，用于人脸检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用 OpenCV 的 Haar 级联检测器进行人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # 取检测到的第一个人脸
            x, y, w, h = faces[0]
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            # 转换为 RGB（Dlib 需要 RGB 格式）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            shape = predictor(img_rgb, dlib_rect)
            face_descriptor = face_reco_model.compute_face_descriptor(img_rgb, shape)
            return face_descriptor
        else:
            print(f"未检测到人脸: {path_img}")
            return None
    except Exception as e:
        print(f"处理图像出错: {path_img}: {e}")
        return None

def get_person_images_with_labels():
    """返回图像路径及其对应的标签（示例：用文件名哈希模拟真实标签）"""
    person_images = []
    for xml_file in os.listdir(person_annotation_dir):
        xml_path = os.path.join(person_annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        has_person = any(obj.find('name').text == 'person' for obj in root.findall('object'))
        if has_person:
            image_name = xml_file.replace('.xml', '.jpg')
            image_path = os.path.join(person_image_dir, image_name)
            label = hash(image_name) % 2  # 模拟标签（0 或 1）
            person_images.append((image_path, label))
    return person_images

# 计算欧氏距离
def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))

# 人脸目标识别，添加阈值判断
def recognize_face(query_descriptor, database, threshold):
    min_distance = float('inf')
    recognized_label = None
    for label, descriptors in database.items():
        for descriptor in descriptors:
            distance = euclidean_distance(query_descriptor, descriptor)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                recognized_label = label
    return recognized_label

def main():
    person_images = get_person_images_with_labels()
    os.makedirs("data", exist_ok=True)

    # 分别保存正样本和负样本特征
    positive_samples = []
    negative_samples = []
    database = {}

    for image_path, label in person_images:
        features = return_128d_features(image_path)
        if features is not None:
            if label == 1:
                positive_samples.append([os.path.basename(image_path), label] + list(features))
            else:
                negative_samples.append([os.path.basename(image_path), label] + list(features))
            if label not in database:
                database[label] = []
            database[label].append(features)
        else:
            print(f"跳过: {image_path}")

    # 保存正样本特征
    with open("data/positive_features.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "label"] + [f"feature_{i}" for i in range(128)])
        for sample in positive_samples:
            writer.writerow(sample)

    # 保存负样本特征
    with open("data/negative_features.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "label"] + [f"feature_{i}" for i in range(128)])
        for sample in negative_samples:
            writer.writerow(sample)

    # 设定阈值，这里可以根据需求调整
    threshold = 0.4

    # 进行人脸目标识别
    for image_path, label in person_images:
        features = return_128d_features(image_path)
        if features is not None:
            recognized_label = recognize_face(features, database, threshold)
            if recognized_label is not None:
                print(f"图像 {image_path} 识别为标签: {recognized_label}")
            else:
                print(f"图像 {image_path} 未找到匹配的人脸")

if __name__ == "__main__":
    main()