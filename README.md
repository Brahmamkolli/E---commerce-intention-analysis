# E-commerce Transaction Intention Analysis

# 1.Busines Problem Understanding

## Notebook summary 

- Notebook file: `E-commerce Transaction Intention Analysis.ipynb`
- Top-level markdown headings found: 12
  - # 1.Busines Problem Understanding
  - # 2.Data Understanding
  - # 3.Data Preprocessing
  - # 4 & 5.Modelling & Evaluation
  - # Logistic Regression
  - # KNN Classifier
  - # Support Vector Machine(SVM)
  - # Decision Tree Classifier
  - # Random Forest Classifier
  - # Ada Boost Classifier

## Data sources

- No explicit `read_csv`/`read_excel` file paths found; notebook may load data from variables or use relative paths not captured.

## Libraries / imports detected

- `from imblearn.over_sampling import SMOTE`
- `from sklearn.ensemble import AdaBoostClassifier`
- `from sklearn.ensemble import GradientBoostingClassifier`
- `from sklearn.ensemble import RandomForestClassifier`
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.metrics import ConfusionMatrixDisplay`
- `from sklearn.metrics import f1_score, classification_report`
- `from sklearn.metrics import roc_auc_score`
- `from sklearn.model_selection import GridSearchCV`
- `from sklearn.model_selection import cross_val_score`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.neighbors import KNeighborsClassifier`
- `from sklearn.preprocessing import LabelEncoder`
- `from sklearn.preprocessing import StandardScaler`
- `from sklearn.svm import SVC`
- `from sklearn.tree import DecisionTreeClassifier`
- `from sklearn.tree import plot_tree`
- `from xgboost import XGBClassifier`
- `import joblib`
- `import matplotlib.pyplot`
- `import numpy`
- `import pandas`
- `import seaborn`

## Packages (suggested for requirements.txt)

```
imbalanced-learn
ipykernel
jupyterlab
matplotlib
numpy
pandas
scikit-learn
seaborn
xgboost
```

## Models & techniques detected in the notebook

- AdaBoostClassifier
- DecisionTreeClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- LogisticRegression
- RandomForestClassifier
- SVC
- XGBClassifier

## Evaluation metrics & artifacts detected

- auc
- classification_report
- f1_score
- roc_auc_score

## Visualizations

- matplotlib
- seaborn

## Reproduction / How to run

1. Install Python 3.8+ (3.9 or 3.10 recommended).
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *If you don't have a `requirements.txt`, use the package list above as a starting point.*
4. Start Jupyter and open the notebook:
   ```bash
   jupyter lab
   ```




