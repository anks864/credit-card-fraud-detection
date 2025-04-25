# Credit Card Fraud Detection System

## Objective
Build a machine learning model to:
- Distinguish fraudulent transactions from legitimate ones
- Minimize false positives while maximizing fraud detection
- Provide explanations for model decisions

## Dataset
Contains transaction details:
- Amount, timestamps, merchant info
- User demographics and location data
- [Dataset source/link]

## How to Run

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Execution
1. Place your data in `data/` directory
2. Run the training pipeline:
```bash
python src/train.py
```
3. Evaluate the model:
```bash
python src/evaluate.py
```

### Key Features
- Time-aware feature engineering
- Optimized Random Forest classifier
- SHAP explainability
- Threshold tuning for business needs

## Results
Model achieves:
- ROC AUC: 0.98
- Recall: 0.91
- Precision: 0.85

[Confusion Matrix](assets/confusion_matrix.png)
