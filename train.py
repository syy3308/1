import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import get_data_loaders
from model import SimplifiedFaceCNN

def train_model():
    train_loader, test_loader = get_data_loaders()

    model = SimplifiedFaceCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss)
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_simplified_face_cnn_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

    return model

if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), 'simplified_face_cnn_model.pth')
    print("模型训练完成并保存")