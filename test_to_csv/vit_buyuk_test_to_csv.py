import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import clip
import torch.nn as nn

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

def final_predict(model, images):
    # Normal tahmin
    outputs = model(images)
    
    # Yatay çevirme ile tahmin
    flipped_images = torch.flip(images, dims=[3])
    outputs_flipped = model(flipped_images)
    
    # İki tahminin ortalamasını al
    final_outputs = (outputs + outputs_flipped) / 2.0
    
    return final_outputs

def main():
    # Yollar
    test_csv_path = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\test.csv"
    test_img_folder = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\test\test"
    output_csv_path = r"C:\Users\Albay\OneDrive\Masaüstü\city_classifier\data\clip_test_sonuclari.csv"
    checkpoint_path = r'C:\Users\Albay\OneDrive\Masaüstü\city_classifier\clip_models\clip_best_acc_95.5714.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test, {device} cihazında çalışacak.")

    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Dataset ve DataLoader
    test_dataset = InferenceDataset(
        csv_file=test_csv_path,
        img_dir=test_img_folder,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model oluştur ve ağırlıkları yükle
    model = CLIPCityClassifier(device=device).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    filenames = []
    preds_list = []

    print("Tahminler yapılıyor...")
    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(test_loader):
            images = images.to(device)
            
            # TTA ve Ensemble tahminleri
            outputs = final_predict(model, images)
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

class CLIPCityClassifier(nn.Module):
    def __init__(self, num_classes=3, device='cuda'):
        super(CLIPCityClassifier, self).__init__()
        self.clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        self.feature_dim = 768
        
        self.clip_model = self.clip_model.float()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
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
            image_features = self.clip_model.encode_image(image).float()
        output = self.classifier(image_features)
        return output

if __name__ == '__main__':
    main()
