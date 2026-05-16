# Customer Churn Prediction - Migration Scripts

This directory contains automated scripts to migrate from the old folder structure (`Customer Churn Prediction`) to the new clean URL structure (`customer-churn-prediction`).

## 📋 Available Scripts

### 1. Python Script (Recommended for all platforms)
```bash
python3 migrate-churn-project.py
```

**Advantages:**
- ✅ Works on Windows, macOS, Linux
- ✅ Better error handling
- ✅ Colored output for clarity
- ✅ No external dependencies

### 2. Bash Script (For Linux/macOS)
```bash
chmod +x migrate-churn-project.sh
./migrate-churn-project.sh
```

**Advantages:**
- ✅ Fast execution
- ✅ Direct system commands
- ✅ Better for large files

## 🚀 Quick Start

### Option A: Automated (Recommended)
```bash
# 1. Run the migration script
python3 migrate-churn-project.py

# 2. Review changes
git status

# 3. Push to GitHub
git push origin main

# 4. Delete old folder
git rm -r "Customer Churn Prediction"
git commit -m "refactor: remove old Customer Churn Prediction folder"
git push origin main
```

### Option B: Manual Step-by-Step
```bash
# 1. Create directories
mkdir -p customer-churn-prediction/{notebooks,models,dataset}

# 2. Copy notebooks (rename them)
cp "Customer Churn Prediction/notebooks/Customer Churn Prediction EN.ipynb" \
   "customer-churn-prediction/notebooks/customer-churn-prediction-en.ipynb"

cp "Customer Churn Prediction/notebooks/Customer Churn Prediction PT_Portuguese_PT.ipynb" \
   "customer-churn-prediction/notebooks/customer-churn-prediction-pt.ipynb"

# 3. Copy models
cp -r "Customer Churn Prediction/models/"* customer-churn-prediction/models/

# 4. Copy README
cp "Customer Churn Prediction/README.md" customer-churn-prediction/README.md

# 5. Stage and commit
git add customer-churn-prediction/
git commit -m "refactor: migrate Customer Churn Prediction to customer-churn-prediction"

# 6. Delete old folder
git rm -r "Customer Churn Prediction"
git commit -m "refactor: remove old Customer Churn Prediction folder"

# 7. Push
git push origin main
```

## 📊 What Gets Migrated

| Item | From | To |
|------|------|-----|
| English Notebook | `Customer Churn Prediction EN.ipynb` | `customer-churn-prediction-en.ipynb` |
| Portuguese Notebook | `Customer Churn Prediction PT_Portuguese_PT.ipynb` | `customer-churn-prediction-pt.ipynb` |
| All Models | `Customer Churn Prediction/models/*` | `customer-churn-prediction/models/*` |
| Dataset | `Customer Churn Prediction/dataset/*` | `customer-churn-prediction/dataset/*` |
| README | Updated with new URLs | ✅ Already done |

## 🔗 URL Changes

**Before (Broken Links on LinkedIn):**
```
https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/Customer%20Churn%20Prediction/notebooks/...
```

**After (Clean URLs):**
```
https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/...
```

## ✅ Verification Checklist

After running the migration script:

- [ ] All notebooks copied and renamed
- [ ] All models copied
- [ ] New folder structure created
- [ ] Git changes staged
- [ ] Commit created
- [ ] Can view at: https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/customer-churn-prediction
- [ ] Old folder deleted
- [ ] Changes pushed to GitHub

## 🐛 Troubleshooting

### Script won't execute (Bash)
```bash
chmod +x migrate-churn-project.sh
```

### Permission denied (Python)
```bash
python3 migrate-churn-project.py  # Use python3 explicitly
```

### Old folder not found
Make sure you're in the repository root directory:
```bash
cd ~/path/to/enioRubens_dataScienceProjects
pwd  # Verify you're in the right place
ls -la "Customer Churn Prediction"  # Should list files
```

### Git errors
Ensure git is installed and you're in a git repository:
```bash
git status  # Should show repository status
```

## 📚 After Migration

### New Project Structure
```
customer-churn-prediction/
├── README.md                              # Project documentation (updated)
├── notebooks/
│   ├── customer-churn-prediction-en.ipynb # English version
│   └── customer-churn-prediction-pt.ipynb # Portuguese version
├── models/
│   ├── LGBMClassifierRocaucOptBalanced
│   ├── XGBClassifierRocaucOptBalanced
│   ├── RandomForestClassifierRocaucOptBalanced
│   └── ... (other model files)
└── dataset/
    └── (your data files)
```

### Shareable Links (Now Work on LinkedIn!)
```
📄 Project: https://github.com/eniorubens/enioRubens_dataScienceProjects/tree/main/customer-churn-prediction
📓 English Notebook: https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/customer-churn-prediction-en.ipynb
📓 Portuguese Notebook: https://github.com/eniorubens/enioRubens_dataScienceProjects/blob/main/customer-churn-prediction/notebooks/customer-churn-prediction-pt.ipynb
```

## 📞 Support

If you encounter issues:

1. Check that you're in the repository root directory
2. Verify `Customer Churn Prediction` folder exists
3. Ensure git is installed (`git --version`)
4. Check file permissions (`ls -la` on Unix systems)
5. Review error messages carefully

---

**Last Updated:** 2026-05-16  
**Author:** Enio Rubens
