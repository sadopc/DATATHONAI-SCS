import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import os

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Kullanılan cihaz: {device}')

# Dizin yolları
TRAIN_DIR = r"data\train\train"
TRAIN_CSV = r"data\train_data.csv"
TEST_DIR = r"data\test"
TEST_CSV = r"data\test.csv"

# Model kayıt dizini
SAVE_DIR = 'efficientnetb4_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Veri seti sınıfı
class CityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'Istanbul': 0, 'Ankara': 1, 'Izmir': 2}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if 'city' in self.data_frame.columns:
            label = self.class_to_idx[self.data_frame.iloc[idx]['city']]
        else:
            label = -1

        if self.transform:
            image = self.transform(image)

        return image, label

# EfficientNetB4 için optimize edilmiş veri dönüşümleri
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # EfficientNetB4 için optimal boyut
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# EfficientNetB4 model tanımı
class EfficientNetB4Classifier(nn.Module):
    def __init__(self):
        super(EfficientNetB4Classifier, self).__init__()
        self.model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        
        # EfficientNetB4'ün son katmanını özelleştirme
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.model(x)

# Veri yükleyicileri
train_dataset = CityDataset(TRAIN_CSV, TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # EfficientNetB4 için küçük batch size

# Validation split
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Early stopping
best_val_acc = 0
patience = 12  # EfficientNetB4 için daha uzun patience
patience_counter = 0

# Model oluşturma
model = EfficientNetB4Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-5)  # Daha düşük learning rate

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=4, verbose=True
)

# Eğitim döngüsü
num_epochs = 15
for epoch in range(num_epochs):
    try:
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Her epoch sonunda modeli kaydet
        model_save_path = os.path.join(SAVE_DIR, f'efficientnetb4_epoch_{epoch+1}_acc_{val_acc:.4f}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model kaydedildi: {model_save_path}')
        
        # En iyi modeli ayrıca kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(SAVE_DIR, f'efficientnetb4_best_acc_{val_acc:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'En iyi model kaydedildi: {best_model_path}')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
            
    except KeyboardInterrupt:
        print("\nEğitim durduruldu!")
        interrupted_model_path = os.path.join(SAVE_DIR, f'efficientnetb4_interrupted_epoch_{epoch+1}_acc_{val_acc:.4f}.pth')
        torch.save(model.state_dict(), interrupted_model_path)
        print(f'Kesintiye uğrayan model kaydedildi: {interrupted_model_path}')
        break

# Test için
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Test verisi için tahmin
test_dataset = CityDataset(TEST_CSV, TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
predictions = predict(model, test_loader)
