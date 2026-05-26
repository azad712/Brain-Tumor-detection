# GitHub Upload Guide for Brain Tumor Detection Project

This guide provides step-by-step instructions for uploading your project to GitHub.

## 📋 Pre-Upload Checklist

Before uploading to GitHub, ensure you have:

- [x] All Python scripts (train.py, predict.py, evaluate.py, etc.)
- [x] README.md with comprehensive documentation
- [x] requirements.txt with all dependencies
- [x] .gitignore file to exclude large files
- [x] LICENSE file
- [x] CONTRIBUTING.md for contributors
- [x] DATASET_SETUP.md for dataset instructions
- [x] data.yaml configuration file

## 🚀 Step-by-Step Upload Process

### Step 1: Prepare Your Local Repository

1. **Navigate to your project directory**:
   ```bash
   cd "C:/Users/Admin/Desktop/CMRIT_Content/STTP Medical AI/brain tumor with BB"
   ```

2. **Initialize Git repository** (if not already done):
   ```bash
   git init
   ```

3. **Configure Git** (first time only):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Step 2: Handle Large Files

**IMPORTANT**: Do NOT upload the dataset directly to GitHub!

The `.gitignore` file already excludes:
- Train/ folder
- Val/ folder
- Test/ folder
- All image files (*.jpg, *.png, etc.)
- Model weights (*.pt files except yolov8*.pt)
- runs/ folder (training outputs)

### Step 3: Create GitHub Repository

1. **Go to GitHub**: https://github.com
2. **Click** "New repository" (+ icon in top right)
3. **Fill in details**:
   - Repository name: `brain-tumor-detection-yolov8`
   - Description: `Brain tumor detection using YOLOv8 with bounding box localization`
   - Visibility: Public or Private
   - **DO NOT** initialize with README (you already have one)
4. **Click** "Create repository"

### Step 4: Add Files to Git

1. **Check what will be committed**:
   ```bash
   git status
   ```

2. **Add all files** (respecting .gitignore):
   ```bash
   git add .
   ```

3. **Verify files to be committed**:
   ```bash
   git status
   ```
   
   You should see:
   - ✅ Python scripts (.py files)
   - ✅ README.md, LICENSE, .gitignore
   - ✅ requirements.txt, data.yaml
   - ✅ Documentation files (.md)
   - ❌ Train/, Val/ folders (excluded)
   - ❌ Image files (excluded)
   - ❌ runs/ folder (excluded)

4. **Commit the files**:
   ```bash
   git commit -m "Initial commit: Brain tumor detection with YOLOv8"
   ```

### Step 5: Push to GitHub

1. **Add remote repository** (replace with your GitHub URL):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-detection-yolov8.git
   ```

2. **Verify remote**:
   ```bash
   git remote -v
   ```

3. **Push to GitHub**:
   ```bash
   git branch -M main
   git push -u origin main
   ```

4. **Enter credentials** when prompted (or use SSH key)

### Step 6: Upload Dataset Separately

Choose one of these options:

#### Option A: Google Drive (Recommended)

1. **Upload dataset to Google Drive**:
   - Create a folder: "Brain Tumor Dataset"
   - Upload Train/ and Val/ folders
   - Right-click → Share → Get link
   - Set to "Anyone with the link can view"

2. **Update README.md** with download link:
   ```markdown
   ## 📊 Dataset
   
   Download the dataset from Google Drive:
   [Brain Tumor Dataset](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)
   
   After downloading, extract to project root:
   ```
   brain-tumor-detection/
   ├── Train/
   └── Val/
   ```
   ```

3. **Commit and push README update**:
   ```bash
   git add README.md
   git commit -m "Add dataset download link"
   git push
   ```

#### Option B: Kaggle Dataset

1. **Create Kaggle account**: https://www.kaggle.com
2. **Upload dataset**: Kaggle → Datasets → New Dataset
3. **Add link to README.md**:
   ```markdown
   ## 📊 Dataset
   
   Download from Kaggle:
   ```bash
   kaggle datasets download -d YOUR_USERNAME/brain-tumor-dataset
   ```
   ```

#### Option C: GitHub Releases (For Model Weights Only)

1. **Train your model first**
2. **Go to your GitHub repository**
3. **Click** "Releases" → "Create a new release"
4. **Tag version**: v1.0.0
5. **Upload files**: best.pt, last.pt
6. **Publish release**

### Step 7: Enhance Your Repository

#### Add Repository Topics

1. Go to your repository on GitHub
2. Click "⚙️ Settings" → "General"
3. Add topics:
   - `deep-learning`
   - `yolov8`
   - `brain-tumor-detection`
   - `medical-imaging`
   - `computer-vision`
   - `pytorch`
   - `object-detection`

#### Add Repository Description

1. Click "⚙️" next to "About"
2. Add description: "Brain tumor detection using YOLOv8 with bounding box localization"
3. Add website (if any)
4. Add topics (as above)

#### Create GitHub Pages (Optional)

1. Go to Settings → Pages
2. Source: Deploy from branch
3. Branch: main, folder: /docs (if you have documentation)
4. Save

### Step 8: Update README with Badges

Add badges to your README.md:

```markdown
# Brain Tumor Detection with YOLOv8

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

[Rest of your README content...]
```

## 📝 Maintaining Your Repository

### Regular Updates

```bash
# Make changes to your code
git add .
git commit -m "Description of changes"
git push
```

### Creating Branches for Features

```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push branch
git push -u origin feature/new-feature

# Create Pull Request on GitHub
```

### Tagging Releases

```bash
# Create tag
git tag -a v1.0.0 -m "First release"

# Push tag
git push origin v1.0.0
```

## 🔒 Security Best Practices

1. **Never commit**:
   - API keys or passwords
   - Personal information
   - Large binary files
   - Temporary files

2. **Use .gitignore** properly

3. **Review before pushing**:
   ```bash
   git diff --cached
   ```

4. **Use environment variables** for sensitive data

## 📊 Repository Structure on GitHub

After upload, your repository should look like:

```
brain-tumor-detection-yolov8/
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── DATASET_SETUP.md
├── GITHUB_UPLOAD_GUIDE.md
├── requirements.txt
├── data.yaml
├── train.py
├── predict.py
├── evaluate.py
├── visualize.py
├── validate_dataset.py
└── yolov8n.pt (optional, small pretrained model)
```

**NOT included** (as per .gitignore):
- Train/ folder
- Val/ folder
- Test/ folder
- runs/ folder
- *.jpg, *.png files
- Large model weights

## ✅ Verification Checklist

After uploading, verify:

- [ ] All Python scripts are present
- [ ] README.md displays correctly
- [ ] .gitignore is working (no large files uploaded)
- [ ] LICENSE file is present
- [ ] requirements.txt is complete
- [ ] Repository description and topics are set
- [ ] Dataset download instructions are clear
- [ ] No sensitive information is exposed

## 🆘 Troubleshooting

### Problem: Files too large

**Solution**: 
```bash
# Remove large files from Git history
git rm --cached large_file.pt
git commit -m "Remove large file"
git push
```

### Problem: Accidentally committed dataset

**Solution**:
```bash
# Remove from Git but keep locally
git rm -r --cached Train/ Val/
git commit -m "Remove dataset from Git"
git push
```

### Problem: Authentication failed

**Solution**:
- Use Personal Access Token instead of password
- GitHub → Settings → Developer settings → Personal access tokens
- Generate new token with repo permissions
- Use token as password when pushing

## 🔗 Useful Commands

```bash
# Check repository status
git status

# View commit history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- filename

# Update from remote
git pull origin main

# Clone your repository
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection-yolov8.git
```

## 📧 Need Help?

- GitHub Documentation: https://docs.github.com
- Git Documentation: https://git-scm.com/doc
- Stack Overflow: https://stackoverflow.com/questions/tagged/git

---

**Congratulations!** 🎉 Your project is now on GitHub!

Share your repository link with others and start collaborating!