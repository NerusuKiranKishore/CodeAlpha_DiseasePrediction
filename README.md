# 🏥 Disease Prediction from Medical Data

> **CodeAlpha Machine Learning Internship — Task 4**

A production-grade, pipeline-structured machine learning project that predicts the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** from patient medical data using classification algorithms.

---

## 📁 Project Structure

```
CodeAlpha_DiseasePrediction/
├── data/                          # Raw datasets (downloaded automatically)
├── src/
│   ├── data/
│   │   └── data_loader.py         # Dataset download + loading
│   ├── features/
│   │   └── feature_engineering.py # Preprocessing + feature engineering
│   ├── models/
│   │   └── trainer.py             # Train, evaluate, compare models
│   ├── visualization/
│   │   └── plots.py               # All plots (ROC, confusion matrix, etc.)
│   └── utils/
│       └── helpers.py             # Logging, saving utilities
├── notebooks/
│   └── exploration.ipynb          # EDA notebook
├── tests/
│   └── test_pipeline.py           # Unit tests
├── outputs/
│   ├── models/                    # Saved .pkl model files
│   ├── plots/                     # Generated plots
│   └── reports/                   # Classification reports
├── main.py                        # Entry point — run full pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

### 3. Run for a specific disease
```bash
python main.py --disease heart
python main.py --disease diabetes
python main.py --disease breast_cancer
```

---

## 📊 Datasets Used

| Disease | Dataset | Source | Samples | Features |
|---|---|---|---|---|
| Heart Disease | Cleveland Heart Disease | UCI ML / sklearn | 303 | 13 |
| Diabetes | Pima Indians Diabetes | sklearn / fetch | 768 | 8 |
| Breast Cancer | Breast Cancer Wisconsin | sklearn.datasets | 569 | 30 |

All datasets are fetched **automatically** via `sklearn.datasets` or direct download — no manual setup needed.

---

## 🤖 Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Cross-Validation (5-fold)

---

## 🧠 Key Features

- **Multi-disease support** — one pipeline, three diseases
- **Production-grade code** — modular, not notebook-spaghetti
- **Auto model selection** — best model saved automatically
- **Rich visualizations** — ROC curves, confusion matrices, feature importance
- **Reproducible** — fixed random seed throughout

---

## 📋 Results Summary

After running `main.py`, results are saved to:
- `outputs/reports/` — Full classification reports (`.txt`)
- `outputs/plots/` — All plots (`.png`)
- `outputs/models/` — Best model per disease (`.pkl`)

---

## 👨‍💻 Author

**Nerusu Kiran Kishore**  
CSE Student, IIIT Sri City  
[GitHub](https://github.com/NerusuKiranKishore) | [LinkedIn](https://www.linkedin.com/in/nerusu-kiran-kishore)

---

*CodeAlpha ML Internship — Task 4 of 4*
