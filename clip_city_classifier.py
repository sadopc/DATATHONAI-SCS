import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import clip  # CLIP modelini import ediyoruz

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Kullanılan cihaz: {device}')

# Dizin yolları
TRAIN_DIR = r"data\train\train"
TRAIN_CSV = r"data\train_data.csv"
TEST_DIR = r"data\test"
TEST_CSV = r"data\test.csv"

# Model kayıt dizini
SAVE_DIR = 'clip_models'
os.makedirs(SAVE_DIR, exist_ok=True)

class CityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'Istanbul': 0, 'Ankara': 1, 'Izmir': 2}
        
        # CLIP için şehir prompt'ları
        self.prompts = {
            'Istanbul': "a photo of Istanbul city, Turkey",
            'Ankara': "a photo of Ankara city, Turkey",
            'Izmir': "a photo of Izmir city, Turkey"
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if 'city' in self.data_frame.columns:
            label = self.class_to_idx[self.data_frame.iloc[idx]['city']]
            prompt = self.prompts[self.data_frame.iloc[idx]['city']]
        else:
            label = -1
            prompt = ""

        if self.transform:
            image = self.transform(image)

        return image, label, prompt

# CLIP için optimize edilmiş veri dönüşümleri
train_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-3, 3)),  # Daha az rotasyon
    transforms.ColorJitter(brightness=0.1),  # Sadece parlaklık
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                       std=[0.26862954, 0.26130258, 0.27577711])
])

test_transform = transforms.Compose([
    transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                       std=[0.26862954, 0.26130258, 0.27577711])
])

class CLIPCityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(CLIPCityClassifier, self).__init__()
        self.clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        self.feature_dim = 768
        
        # CLIP modelini float32'ye dönüştür
        self.clip_model = self.clip_model.float()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        with torch.no_grad():
            # Görüntü özelliklerini float32'ye dönüştür
            image_features = self.clip_model.encode_image(image).float()
        
        output = self.classifier(image_features)
        return output

# Veri yükleyicileri
train_dataset = CityDataset(TRAIN_CSV, TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation split
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model oluşturma
model = CLIPCityClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 
                            lr=3e-5,  # Daha yüksek learning rate
                            weight_decay=0.01)

# Önce num_epochs'u tanımla
num_epochs = 30

# Sonra scheduler'ı tanımla
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,  # Daha yüksek max learning rate
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

# Early stopping
best_val_acc = 0
patience = 10
patience_counter = 0

# Eğitim döngüsü
for epoch in range(num_epochs):
    try:
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels, _ in train_loader:
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
            for images, labels, _ in val_loader:
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
        
        # Model kaydetme
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(SAVE_DIR, f'clip_best_acc_{val_acc:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'En iyi model kaydedildi: {best_model_path}')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
            
    except KeyboardInterrupt:
        print("\nEğitim durduruldu!")
        interrupted_model_path = os.path.join(SAVE_DIR, f'clip_interrupted_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), interrupted_model_path)
        print(f'Kesintiye uğrayan model kaydedildi: {interrupted_model_path}')
        break

# Test fonksiyonu
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Test verisi için tahmin
test_dataset = CityDataset(TEST_CSV, TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
predictions = predict(model, test_loader)

def final_predict(model, image):
    # Ensemble tahminleri için modelleri oluştur
    models = {
        'model1': model,  # Ana model
        'model2': model   # Aynı modelin kopyası (TTA için)
    }
    
    # Ensemble tahminleri
    ensemble_predictions = []
    for m in models.values():
        pred = m(image)
        ensemble_predictions.append(pred)
    ensemble_pred = torch.mean(torch.stack(ensemble_predictions), dim=0)
    
    # TTA tahminleri
    tta_pred = tta_predict(model, image)  # Direkt modeli gönderiyoruz
    
    # İki tahmini birleştir
    final_pred = (ensemble_pred + tta_pred) / 2
    return final_pred

def tta_predict(model, image, n_augments=10):
    model.eval()  # Modeli değerlendirme moduna al
    predictions = []
    
    # Temel dönüşümler
    basic_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02))
    ])
    
    with torch.no_grad():
        # Orijinal görüntü tahmini
        predictions.append(model(image))
        
        # Augmented görüntü tahminleri
        for _ in range(n_augments):
            # Görüntüyü kopyala
            aug_image = image.clone()
            
            # Dönüşümleri uygula
            aug_image = basic_transforms(aug_image)
            
            # Tahmin yap
            pred = model(aug_image)
            predictions.append(pred)
    
    # Tüm tahminlerin ortalamasını al
    final_prediction = torch.mean(torch.stack(predictions), dim=0)
    
    return final_prediction