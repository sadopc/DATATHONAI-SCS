### **1. Modellerin Çalıştırılması**

Aşağıda listelenen modellerin her birini çalıştırmanız gerekmektedir:

- `clip_city_classifier`
- `convnext_city_classifier`
- `efficientnet_b4_city_classifier`
- `efficientnet_b7_city_classifier`
- `efficientnet_v2_m_city_classifier`
- `resnet50_city_classifier`

**Adımlar:**


### **1. **Modeli Çalıştırma:**
   Her bir modelin çalıştırılması için ilgili Python betiğini (`.py` dosyası) çalıştırın. (Dizinlere dokunmassaniz oldugu hali ile calisiyor olmali.)
   Örneğin:

   python clip_city_classifier.py
   python convnext_city_classifier.py
   python efficientnet_b4_city_classifier.py
   python efficientnet_b7_city_classifier.py
   python efficientnet_v2_m_city_classifier.py
   python resnet50_city_classifier.py

   Buradaki her bir kod yeni bir model  (`.pth` uzantili)  olusturacaktir.
---

### **2. Test Verilerinin CSV'e Dönüştürülmesi**

Modeller çalıştırıldıktan sonra, her model icin test yapip, tahmin sonuçlarını CSV formatına dönüştürmeniz gerekmektedir. Bunun için aşağıdaki klasörlerde bulunan dönüşüm betiklerini çalıştırın 
(butun modeller icin model.pth'leri gostermeniz gerekiyor):

- `test_to_csv\convnext_test_to_csv`
- `test_to_csv\efficientnet_b4_test_to_csv`
- `test_to_csv\efficientnet_b7_test_to_csv`
- `test_to_csv\efficientnet_v2_m_test_to_csv`
- `test_to_csv\resnet_50_test_to_csv`
- `test_to_csv\vit_buyuk_test_to_csv`

**Adımlar:**

1. **Dönüşüm Betiklerinin Çalıştırılması:**
   Her klasörde bulunan dönüşüm betiklerini (örneğin, `convnext_test_to_csv.py`) çalıştırarak testleri yapin ve tahmin sonuçlarını CSV formatına dönüştürün. 
   (Her bir model icin yapmaniz gerekmektedir)

   cd test_to_csv/convnext_test_to_csv (Hem testleri yapip hemde csv ye donusturuyor)
   
   
   # Diğer klasörler için de aynı işlemi tekrarlayın

2. **CSV Dosyalarının Kontrol Edilmesi:**
   Dönüşüm işlemi tamamlandıktan sonra, her klasörde `model_test_sonuclari.csv` gibi isimlendirilmiş CSV dosyalarının oluşturulduğunu doğrulayın.

---

### **3. CSV Dosyalarının Karşılaştırılması ve Sonuçların Birleştirilmesi**

Tüm modellerden elde edilen CSV dosyalarını karşılaştırarak en çok çıkan tahminleri almak için aşağıdaki adımları izleyin.

**Adımlar:**

1. **CSV Karşılaştırma Betiğinin Çalıştırılması:**
   `csv_karsilastirma.py` adlı betiği çalıştırarak tüm CSV dosyalarını karşılaştırın ve en çok çıkan tahminleri belirleyin.

   python csv_karsilastirma.py

2. **Sonuçların İncelenmesi:**
   Betik çalıştıktan sonra, en çok tekrar eden tahminlerin yer aldığı yeni bir CSV dosyası oluşturulacaktır 
   (`ensemble_sonuclar.csv`). Bu dosyayı inceleyerek yarışmaya göndermeye hazır hale getirin.

