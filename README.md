# Amazon Hackathon – Price Prediction Challenge  

This repository contains my solution for the **Amazon Hackathon 2025**, where the goal was to predict product prices based on structured features, product descriptions (text), and product images.  

---

## 📌 Problem Statement  

Participants were provided with:
- **Structured data** (e.g., weight, unit, brand, category)  
- **Text data** (titles, bullet points, descriptions)  
- **Image data** (product images)  

The task was to build a model that accurately predicts the **price** of each product.  
The final evaluation metric was **SMAPE (Symmetric Mean Absolute Percentage Error)**.

---

## 📂 Project Structure  

hackathon-data/
│
├── student_resource/
│ ├── dataset/ # original train/test CSVs (ignored in git)
│ ├── features/ # engineered features (ignored in git)
│ ├── embeddings/ # pretrained embeddings (ignored in git)
│ ├── notebooks/ # Jupyter notebooks for EDA & training
│ │ ├── 00_eda.ipynb
│ │ ├── 01_feature_engineering.ipynb
│ │ ├── 02_download_images.ipynb
│ │ ├── 03_image_embeddings.ipynb
│ │ ├── 04_model_training.ipynb
│ │ └── 05_final_submission.ipynb
│ └── src/ # helper scripts (if any)
│
├── submissions/
│ └── submission_lgb_tuned.csv # ✅ final tuned submission
│
├── requirements.txt # dependencies
├── README.md # project documentation
└── .gitignore # ignores data, models, caches


---

## ⚙️ Approach  

1. **Data Preprocessing**  
   - Cleaned structured data (`train.csv`, `test.csv`)  
   - Handled missing values and categorical encodings  

2. **Feature Engineering**  
   - TF-IDF embeddings from product descriptions (200-dim)  
   - Image embeddings using pre-trained ResNet (256-dim PCA compressed)  
   - Combined with structured features  

3. **Modeling**  
   - Main model: **LightGBM**  
   - 5-Fold Cross Validation  
   - Tuned hyperparameters (learning rate, num_leaves, min_data_in_leaf, L1/L2 regularization)  

4. **Ensembling Attempts**  
   - Tried LightGBM + XGBoost + CatBoost  
   - Final submission used **tuned LightGBM** (best stable result)  

---

## 📊 Results  

- **Best CV SMAPE:** ~51.8  
- **Final Submission File:** `submission_lgb_tuned.csv`  

---

## 🚀 How to Run  

1. Clone this repository  
   ```bash
   git clone https://github.com/<your-username>/amazon-hackathon.git
   cd amazon-hackathon
2. Install dependencies
''' bash
pip install -r requirements.txt


3. Open notebooks for exploration
'''bash
jupyter notebook

## 🛠️ Tech Stack

Python 3.10

Pandas, NumPy, Scikit-learn

LightGBM, XGBoost, CatBoost

Jupyter Notebook

## 📌 Notes

Large data files (dataset/, features/, embeddings/, images/) are not included in this repository due to size.

Only the final tuned submission CSV is provided.