import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

# ---------------------------- Judul ----------------------------
st.title("Analisis Obesitas dan Pemodelan ML")
st.write("Dataset: ObesityDataSet.csv")

# ---------------------------- Load dataset ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv('ObesityDataSet.csv')

df = load_data()
st.subheader("Data Asli")
st.dataframe(df.head())

# ---------------------------- Preprocessing ----------------------------
st.subheader("Pembersihan Data")
st.write("Jumlah Missing Values:")
st.write(df.isnull().sum())
st.write(f"Jumlah Duplikasi: {df.duplicated().sum()}")

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
if 'NObeyesdad' in cat_cols:
    cat_cols.remove('NObeyesdad')

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

target_encoder = LabelEncoder()
df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])

st.success("Data selesai dibersihkan.")
st.dataframe(df.head())

# ---------------------------- SMOTE dan Split ----------------------------
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

@st.cache_data
def smote_data(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)

X_res, y_res = smote_data(X, y)
st.subheader("Distribusi Kelas Setelah SMOTE")
st.bar_chart(pd.Series(y_res).value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ---------------------------- Evaluasi Model ----------------------------
st.subheader("Evaluasi Model – Baseline dan Tuned")

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te, label='Baseline'):
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    st.markdown(f"### 🔍 Visualisasi Confusion Matrix – {label}")

    selected_models = st.multiselect(
        f"Pilih model yang ingin ditampilkan untuk {label.lower()}:",
        list(models_dict.keys()),
        default=list(models_dict.keys()),
        key=label
    )

    for name, mdl in models_dict.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)

        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_te, y_pred))
        results['Precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
        results['Recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
        results['F1 Score'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

        if name in selected_models:
            with st.expander(f"📊 Confusion Matrix: {name} ({label})"):
                cm = confusion_matrix(y_te, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues', ax=ax, colorbar=False)
                st.pyplot(fig)
                plt.close(fig)

    return pd.DataFrame(results)

# Baseline
baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)

# Tuning
param_grids = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    'KNN': {
        'n_neighbors': list(range(3, 11, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

tuned_models = {}
best_params = {}

for name, mdl in baseline_models.items():
    search = (RandomizedSearchCV(mdl, param_grids[name], n_iter=10, cv=5, scoring='f1_weighted', n_jobs=-1, random_state=42)
              if name == 'Random Forest' else
              GridSearchCV(mdl, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1))

    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    best_params[name] = search.best_params_

    st.write(f"Best Params untuk {name}:", best_params[name])

tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')

# ---------------------------- Visualisasi Performa ----------------------------
baseline_metrics['Tipe'] = 'Baseline'
tuned_metrics['Tipe'] = 'Tuned'
combined_metrics = pd.concat([baseline_metrics, tuned_metrics], ignore_index=True)

st.subheader("📈 Perbandingan Performa Model")
metrics_melted = combined_metrics.melt(id_vars=['Model', 'Tipe'], var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted)
plt.title('Perbandingan Baseline vs Tuned')
plt.ylim(0, 1.05)
st.pyplot(plt.gcf())

# ---------------------------- Kesimpulan ----------------------------
st.subheader("Kesimpulan")

best_baseline = baseline_metrics.loc[baseline_metrics['F1 Score'].idxmax()]
best_tuned = tuned_metrics.loc[tuned_metrics['F1 Score'].idxmax()]
improvement = best_tuned['F1 Score'] - best_baseline['F1 Score']

st.write(f"Model terbaik sebelum tuning: **{best_baseline['Model']}** (F1 Score = {best_baseline['F1 Score']:.4f})")
st.write(f"Model terbaik setelah tuning: **{best_tuned['Model']}** (F1 Score = {best_tuned['F1 Score']:.4f})")
st.write(f"Peningkatan F1 Score: **{improvement:.4f}**")

if improvement > 0:
    st.success("Hyperparameter tuning berhasil meningkatkan performa model.")
else:
    st.warning("Tuning tidak memberikan peningkatan signifikan.")
