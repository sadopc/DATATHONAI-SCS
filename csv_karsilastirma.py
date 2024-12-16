import pandas as pd
import torch
from collections import Counter

# GPU kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# CSV dosyalarını oku
convnext_df = pd.read_csv('data/convnext_test_sonuclari.csv')
efficientnet_df = pd.read_csv('data/efficientnet_v2_m_test_sonuclari.csv') 
efficientnetb4_df = pd.read_csv('data/efficientnetb4_test_sonuclari.csv')
efficientnetb7_df = pd.read_csv('data/efficientnet_b7_sonuclar.csv')
resnet50_df = pd.read_csv('data/resnet50_test_sonuclari.csv')
clip_df = pd.read_csv('data/clip_test_sonuclari.csv')

# Şehir isimlerini sayısal değerlere dönüştür
sehir2idx = {'Istanbul': 0, 'Ankara': 1, 'Izmir': 2}
idx2sehir = {0: 'Istanbul', 1: 'Ankara', 2: 'Izmir'}

# DataFrame'leri tensor'lara dönüştür
def df_to_tensor(df):
    return torch.tensor([sehir2idx[city] for city in df['city'].values], device=device)

# Tüm tahminleri tensor'lara dönüştür
tahminler = torch.stack([
    df_to_tensor(convnext_df),
    df_to_tensor(efficientnet_df),
    df_to_tensor(efficientnetb4_df),
    df_to_tensor(efficientnetb7_df),
    df_to_tensor(resnet50_df),
    df_to_tensor(clip_df)
])

# Her görüntü için en çok tekrar eden sınıfı bul
en_cok_tekrar = torch.mode(tahminler, dim=0).values
en_cok_tekrar = en_cok_tekrar.cpu().numpy()

# Sayısal değerleri şehir isimlerine çevir
sonuc_sehirler = [idx2sehir[idx] for idx in en_cok_tekrar]

# Sonuçları DataFrame'e kaydet
sonuc_df = pd.DataFrame({
    'filename': convnext_df['filename'],
    'city': sonuc_sehirler
})

# CSV'ye kaydet
sonuc_df.to_csv('data/ensemble_sonuclar.csv', index=False)
print("Ensemble sonuçları başarıyla kaydedildi!")
