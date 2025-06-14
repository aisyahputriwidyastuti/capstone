import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🎯 UAS Capstone Bengkel Koding - Klasifikasi Obesitas")

# === Load Dataset ===
df = pd.read_csv("ObesityDataSet.csv")

# === Preprocessing ===
st.subheader("📊 Preprocessing Data")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('NObeyesdad')
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
target_encoder = LabelEncoder()
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# SMOTE
st.write("Distribusi kelas sebelum SMOTE:", y.value_counts().to_dict())
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
st.write("Distribusi kelas setelah SMOTE:", pd.Series(y_res).value_counts().to_dict())

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# === Baseline Models ===
st.subheader("📈 Baseline Model Evaluation")

baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te, label='Baseline'):
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    for name, mdl in models_dict.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_te, y_pred))
        results['Precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        results['Recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        results['F1 Score'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'Confusion Matrix – {name} ({label})')
        st.pyplot(fig)

    return pd.DataFrame(results)

baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)

# === Hyperparameter Tuning ===
st.subheader("🔧 Hyperparameter Tuning")

param_grids = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

tuned_models = {}
best_params = {}

for name, mdl in baseline_models.items():
    st.text(f"Tuning model: {name}")
    if name == 'Random Forest':
        search = RandomizedSearchCV(mdl, param_grids[name], n_iter=10, cv=3, scoring='f1_weighted', n_jobs=-1, random_state=42)
    else:
        search = GridSearchCV(mdl, param_grids[name], cv=3, scoring='f1_weighted', n_jobs=-1)
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    best_params[name] = search.best_params_
    st.write(f"Best params {name}: {search.best_params_}")

# === Evaluation after tuning ===
st.subheader("📊 Evaluasi Model Setelah Tuning")
tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')

# === Visualisasi Perbandingan ===
st.subheader("📉 Perbandingan Performa Model")
baseline_metrics['Tipe'] = 'Baseline'
tuned_metrics['Tipe'] = 'Tuned'
combined = pd.concat([baseline_metrics, tuned_metrics], ignore_index=True)
melted = combined.melt(id_vars=["Model", "Tipe"], var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ci=None, ax=ax)
plt.title("Perbandingan Performa Model – Baseline vs Tuned")
plt.ylim(0, 1.05)
st.pyplot(fig)

# === Kesimpulan ===
st.subheader("✅ Kesimpulan")
best_base = baseline_metrics.loc[baseline_metrics["F1 Score"].idxmax()]
best_tuned = tuned_metrics.loc[tuned_metrics["F1 Score"].idxmax()]
improvement = best_tuned["F1 Score"] - best_base["F1 Score"]

st.write(f"Model terbaik (baseline): {best_base['Model']} (F1 = {best_base['F1 Score']:.4f})")
st.write(f"Model terbaik (tuned): {best_tuned['Model']} (F1 = {best_tuned['F1 Score']:.4f})")
st.write(f"Peningkatan F1 Score: {improvement:.4f}")

if improvement > 0:
    st.success("🎉 Hyperparameter tuning berhasil meningkatkan performa model!")
else:
    st.warning("⚠️ Hyperparameter tuning tidak meningkatkan performa secara signifikan.")

