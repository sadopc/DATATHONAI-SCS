import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Etiket sözlüğü
idx2label = {0: 'Istanbul', 1: 'Ankara', 2: 'Izmir'}

class InferenceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ConvNeXt model sınıfı
class ConvNextClassifier(nn.Module):
    def __init__(self):
        super(ConvNextClassifier, self).__init__()
        self.model = models.convnext_tiny(weights='IMAGENET1K_V1')
        
        self.model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(p=0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.model(x)

def main():
    test_csv_path = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\test.csv"
    test_img_folder = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\test\test"
    output_csv_path = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\convnext_test_sonuclari.csv"
    checkpoint_path = r'C:\Users\Albay\OneDrive\Masaüstü\city_classifier\convnext_models\test1.pth'  # En iyi modelin yolunu buraya yazın

    # ConvNeXt için transform
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = InferenceDataset(
        csv_file=test_csv_path,
        img_dir=test_img_folder,
        transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test, {device} cihazında çalışacak.")

    # Model oluştur ve ağırlıkları yükle
    model = ConvNextClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    filenames = []
    preds_list = []

    print("Tahminler yapılıyor...")
    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            
            filenames.extend(names)
            preds_list.extend(preds.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(test_loader)} işlendi")

    # Tahminleri şehir isimlerine çevir
    pred_labels = [idx2label[p] for p in preds_list]

    # Sonuçları DataFrame'e kaydet
    df_test = pd.DataFrame({
        'filename': filenames,
        'city': pred_labels
    })

    # CSV'ye kaydet
    df_test.to_csv(output_csv_path, index=False)
    print(f"Tahminler '{output_csv_path}' dosyasına kaydedildi.")
    print(f"Toplam {len(df_test)} görüntü sınıflandırıldı.")

if __name__ == '__main__':
    main()
