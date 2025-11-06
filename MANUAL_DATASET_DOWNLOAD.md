# Manual Dataset Download Guide

Since automatic downloads are failing due to URL restrictions, here's how to manually download the retinal datasets:

## 📥 **Manual Download Instructions**

### **1. DRIVE Dataset (Digital Retinal Images for Vessel Extraction)**
**Website**: https://drive.grand-challenge.org/site/competitions/drive/
- Click on "Download" or "Data" section
- Download the training and test images with vessel annotations
- Save as: `DRIVE.zip`
- Extract to: `dataset/drive/`

**Alternative**: https://www.isi.uu.nl/Research/Databases/DRIVE/
- Look for the download section
- Contains 40 images (20 training, 20 test) with manual annotations

### **2. STARE Dataset (STructured Analysis of the Retina)**
**Website**: https://cecas.clemson.edu/~ahoover/stare/
- Navigate to the "Database" or "Download" section
- Download the retinal images and vessel annotations
- Save as: `STARE.zip` or `stare-images.tar.gz`
- Extract to: `dataset/stare/`

**Alternative**: https://www.clemson.edu/cecas/research/labs/vipl/downloads/
- Look for STARE dataset downloads
- Contains 20 images with expert annotations

### **3. MESSIDOR Dataset (Diabetic Retinopathy Images)**
**Website**: https://www.adcis.net/en/third-party/messidor/
- Register for free account if required
- Download the dataset from the download section
- Save as: `Messidor.zip`
- Extract to: `dataset/messidor/`

**Alternative**: Search for "MESSIDOR dataset download" on Google
- Multiple academic mirrors available
- Contains 1200 retinal images

### **4. CHASE_DB1 Dataset**
**Website**: https://blogs.kingston.ac.uk/retinal/chasedb1/
- Download the retinal images and vessel annotations
- Save as: `CHASE_DB1.zip`
- Extract to: `dataset/chase_db1/`

**Alternative**: https://www.kaggle.com/datasets/andrewmvd/chasedb1
- Requires Kaggle account
- Download via Kaggle API or web interface

## 🔄 **Alternative Sources**

### **Kaggle Datasets (Easiest Option):**
```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets
kaggle datasets download -d andrewmvd/drive-digital-retinal-images-for-vessel-extraction
kaggle datasets download -d kmader/stare-retina-dataset
kaggle datasets download -d googleai/messidor2
kaggle datasets download -d andrewmvd/chasedb1
```

### **Academic Repositories:**
1. **IEEE DataPort**: https://ieee-dataport.org/
2. **Figshare**: https://figshare.com/
3. **Zenodo**: https://zenodo.org/
4. **Mendeley Data**: https://data.mendeley.com/

### **University Research Pages:**
1. **University of Groningen**: https://www.isi.uu.nl/Research/Databases/
2. **University of Iowa**: https://medicine.uiowa.edu/eye/
3. **Clemson University**: https://www.clemson.edu/cecas/departments/bioe/research/

## 📁 **Expected Directory Structure**

After downloading and extracting, your `dataset/` folder should look like:

```
dataset/
├── drive/
│   ├── training/
│   │   ├── images/
│   │   └── 1st_manual/
│   └── test/
│       ├── images/
│       └── 1st_manual/
├── stare/
│   ├── images/
│   └── labels/
├── messidor/
│   ├── Base11/
│   ├── Base12/
│   ├── Base13/
│   └── Base14/
├── chase_db1/
│   ├── images/
│   └── masks/
├── train/          # Your existing synthetic images
└── test/           # Your existing test images
```

## ✅ **Verification Steps**

After downloading, verify your datasets:

```python
# Run this to check if datasets are properly loaded
python test_real_data.py
```

Expected output should show:
- ✅ Successfully loaded X images for each dataset
- ✅ Preprocessing working
- ✅ Feature extraction working

## 🚀 **Next Steps After Download**

Once datasets are downloaded:

```bash
# 1. Test the datasets
python test_real_data.py

# 2. Run real data integration
python real_data_integration.py

# 3. Evaluate performance
python src/ml_model.py
```

## 📞 **If You Need Help**

If you're having trouble with any dataset:

1. **DRIVE**: Most commonly available, try multiple mirrors
2. **STARE**: Check Clemson University website
3. **MESSIDOR**: May require registration
4. **CHASE_DB1**: Available on multiple platforms

## 💡 **Quick Start with Synthetic Data**

While downloading real datasets, you can continue development with synthetic data:

```bash
# Your existing synthetic pipeline still works
python src/ml_model.py
python demo.py
```

The synthetic data approach gives you 92%+ accuracy and is perfect for:
- Algorithm development
- Pipeline testing
- Feature validation
- Model architecture design

## 🎯 **Success Criteria**

✅ **Real Data Integration Complete** when:
- At least 1-2 datasets downloaded successfully
- `python test_real_data.py` shows successful loading
- `python real_data_integration.py` completes without errors
- Model accuracy improves from synthetic baseline

---

**Note**: Dataset URLs frequently change. If links don't work, search for "[Dataset Name] retinal vessel segmentation download" on Google Scholar or ResearchGate.
