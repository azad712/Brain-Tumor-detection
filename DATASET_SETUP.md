# Dataset Setup Guide

This guide explains how to set up the brain tumor dataset for this project.

## 📦 Dataset Storage Options

### Option 1: Git LFS (Recommended for Small Datasets < 2GB)

If your dataset is relatively small, you can use Git Large File Storage (LFS):

1. **Install Git LFS**:
   ```bash
   # Windows (using Chocolatey)
   choco install git-lfs
   
   # Mac
   brew install git-lfs
   
   # Linux
   sudo apt-get install git-lfs
   ```

2. **Initialize Git LFS**:
   ```bash
   git lfs install
   ```

3. **Track large files**:
   ```bash
   git lfs track "*.jpg"
   git lfs track "*.png"
   git lfs track "*.pt"
   git add .gitattributes
   ```

4. **Commit and push**:
   ```bash
   git add Train/ Val/
   git commit -m "Add dataset"
   git push
   ```

### Option 2: External Storage (Recommended for Large Datasets)

For datasets larger than 2GB, use external storage services:

#### A. Google Drive
1. Upload your dataset to Google Drive
2. Make the folder publicly accessible or share with specific users
3. Add download instructions to README.md

#### B. Kaggle Datasets
1. Create a Kaggle account
2. Upload dataset to Kaggle Datasets
3. Provide Kaggle dataset link in README.md

#### C. Hugging Face Hub
1. Create a Hugging Face account
2. Upload dataset to Hugging Face Datasets
3. Provide dataset link in README.md

#### D. AWS S3 / Azure Blob Storage
1. Upload to cloud storage
2. Generate public or signed URLs
3. Provide download script

### Option 3: GitHub Releases (For Model Weights)

Store trained model weights in GitHub Releases:

1. Train your model
2. Go to your GitHub repository
3. Click "Releases" → "Create a new release"
4. Upload `best.pt` and `last.pt` files
5. Publish release

## 📁 Required Dataset Structure

Your dataset should follow this structure:

```
brain-tumor-detection/
├── Train/
│   ├── Glioma/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   ├── Meningioma/
│   │   ├── images/
│   │   └── labels/
│   ├── No Tumor/
│   │   ├── images/
│   │   └── labels/
│   └── Pituitary/
│       ├── images/
│       └── labels/
└── Val/
    ├── Glioma/
    ├── Meningioma/
    ├── No Tumor/
    └── Pituitary/
```

## 📝 Label Format

Each `.txt` label file should contain bounding box annotations in YOLO format:

```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer (0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized box dimensions (0-1)

Example:
```
0 0.5 0.5 0.3 0.4
```

## 🔧 Setup Instructions for Users

### If Dataset is on Google Drive:

1. **Download the dataset**:
   ```bash
   # Install gdown
   pip install gdown
   
   # Download from Google Drive
   gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID
   ```

2. **Extract and organize**:
   ```bash
   # If compressed
   unzip dataset.zip
   
   # Move to project directory
   mv dataset/Train ./Train
   mv dataset/Val ./Val
   ```

### If Dataset is on Kaggle:

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle credentials**:
   - Go to Kaggle → Account → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

3. **Download dataset**:
   ```bash
   kaggle datasets download -d username/brain-tumor-dataset
   unzip brain-tumor-dataset.zip
   ```

### If Dataset is on Hugging Face:

1. **Install Hugging Face Hub**:
   ```bash
   pip install huggingface_hub
   ```

2. **Download dataset**:
   ```python
   from huggingface_hub import snapshot_download
   
   snapshot_download(
       repo_id="username/brain-tumor-dataset",
       repo_type="dataset",
       local_dir="./dataset"
   )
   ```

## ✅ Verify Dataset

After downloading, verify the dataset structure:

```bash
python validate_dataset.py
```

This will check:
- Directory structure
- Image and label file counts
- Label format validity
- Class distribution

## 📊 Dataset Statistics

Run validation to see dataset statistics:

```bash
python validate_dataset.py
```

Expected output:
```
DATASET STATISTICS
======================================================================

TRAINING SET:
----------------------------------------------------------------------
Class                Images     Labels     Annotations    
----------------------------------------------------------------------
Glioma               826        826        826            
Meningioma           822        822        822            
No Tumor             395        395        395            
Pituitary            827        827        827            
----------------------------------------------------------------------
TOTAL                2870       2870       2870           

VALIDATION SET:
----------------------------------------------------------------------
Class                Images     Labels     Annotations    
----------------------------------------------------------------------
Glioma               100        100        100            
Meningioma           115        115        115            
No Tumor             105        105        105            
Pituitary            74         74         74             
----------------------------------------------------------------------
TOTAL                394        394        394            
```

## 🚨 Important Notes

1. **Do NOT commit large datasets to Git** without Git LFS
2. **Always add dataset folders to .gitignore** if not using Git LFS
3. **Provide clear download instructions** in README.md
4. **Include dataset license information** if applicable
5. **Verify dataset integrity** after download

## 📄 Sample Download Script

Create a `download_dataset.py` script for users:

```python
import os
import gdown
import zipfile

def download_dataset():
    """Download and extract brain tumor dataset"""
    
    # Google Drive file ID
    file_id = "YOUR_FILE_ID_HERE"
    output = "dataset.zip"
    
    print("Downloading dataset from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("Cleaning up...")
    os.remove(output)
    
    print("✅ Dataset downloaded and extracted successfully!")
    print("Run 'python validate_dataset.py' to verify the dataset.")

if __name__ == "__main__":
    download_dataset()
```

## 🔗 Useful Links

- [Git LFS Documentation](https://git-lfs.github.com/)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [Google Drive API](https://developers.google.com/drive)

---

For questions about dataset setup, please open an issue on GitHub.