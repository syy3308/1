import torch
import torch.nn as nn
from data_loader import get_data_loaders
from model import ImprovedFaceCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model():
    # Define the device: use 'cuda' if available, otherwise use 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader = get_data_loaders()
    model = ImprovedFaceCNN()

    try:
        # Load the model and move it to the defined device
        checkpoint = torch.load('improved_face_cnn_model.pth', map_location=device)
        model_dict = model.state_dict()
        # 过滤掉不匹配的键
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    except Exception as e:
        print(f"加载模型状态字典时出错: {e}")
        return

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_loader:
            # Move features and labels to the defined device
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")


if __name__ == "__main__":
    evaluate_model()

