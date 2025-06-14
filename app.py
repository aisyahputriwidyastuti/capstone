import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)

# Set page config
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("UAS Capstone Bengkel Koding - Klasifikasi Obesitas")

# Cache functions untuk performa yang lebih baik
@st.cache_data
def load_and_preprocess_data():
    """Load dan preprocess data dengan error handling"""
    try:
        # Load dataset
        df = pd.read_csv("ObesityDataSet.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File 'ObesityDataSet.csv' tidak ditemukan!")
        st.info("Pastikan file CSV berada di direktori yang sama dengan script.")
        return None
    except Exception as e:
        st.error(f"❌ Error saat memuat data: {str(e)}")
        return None

def remove_outliers_iqr(data, column):
    """Remove outliers menggunakan IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te, label='Baseline'):
    """Evaluasi model dengan error handling"""
    results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    for name, mdl in models_dict.items():
        try:
            # Fit model
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)

            # Calculate metrics
            results['Model'].append(name)
            results['Accuracy'].append(accuracy_score(y_te, y_pred))
            results['Precision'].append(precision_score(y_te, y_pred, average='weighted', zero_division=0))
            results['Recall'].append(recall_score(y_te, y_pred, average='weighted', zero_division=0))
            results['F1 Score'].append(f1_score(y_te, y_pred, average='weighted', zero_division=0))

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_te, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Confusion Matrix – {name} ({label})')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Tutup figure untuk menghemat memory
            
        except Exception as e:
            st.error(f"❌ Error saat evaluasi model {name}: {str(e)}")
            continue

    return pd.DataFrame(results)

# === Main Application ===
def main():
    # Load data
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Display basic info
    st.subheader("📊 Informasi Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Baris", df.shape[0])
    with col2:
        st.metric("Jumlah Kolom", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # === Preprocessing ===
    st.subheader("🔄 Preprocessing Data")
    
    # Remove duplicates and missing values
    original_rows = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    st.write(f"Baris yang dihapus (duplikat/missing): {original_rows - len(df)}")
    
    # Remove outliers
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    original_rows_outlier = len(df)
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    st.write(f"Baris yang dihapus (outliers): {original_rows_outlier - len(df)}")
    st.write(f"Dataset final: {len(df)} baris")
    
    # Check if target column exists
    if 'NObeyesdad' not in df.columns:
        st.error("❌ Kolom target 'NObeyesdad' tidak ditemukan!")
        return
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'NObeyesdad' in cat_cols:
        cat_cols.remove('NObeyesdad')
    
    # Store original encoders for potential use
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])
    
    # Show target distribution
    st.write("🎯 Distribusi kelas target:")
    target_dist = df['NObeyesdad'].value_counts().sort_index()
    st.bar_chart(target_dist)
    
    # Prepare features and target
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    # === SMOTE ===
    st.subheader("⚖️ SMOTE - Balancing Classes")
    
    st.write("Distribusi kelas sebelum SMOTE:")
    before_smote = y.value_counts().sort_index().to_dict()
    st.write(before_smote)
    
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        st.write("Distribusi kelas setelah SMOTE:")
        after_smote = pd.Series(y_res).value_counts().sort_index().to_dict()
        st.write(after_smote)
        
    except Exception as e:
        st.error(f"❌ Error saat SMOTE: {str(e)}")
        return
    
    # === Standardization ===
    st.subheader("📏 Standardisasi Fitur")
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        st.success("✅ Standardisasi berhasil")
        
    except Exception as e:
        st.error(f"❌ Error saat standardisasi: {str(e)}")
        return
    
    # === Split data ===
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_res, test_size=0.2, stratify=y_res, random_state=42
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Set", len(X_train))
        with col2:
            st.metric("Test Set", len(X_test))
            
    except Exception as e:
        st.error(f"❌ Error saat split data: {str(e)}")
        return
    
    # === Baseline Models ===
    st.subheader("🚀 Baseline Model Evaluation")
    
    baseline_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    with st.spinner("Evaluasi baseline models..."):
        baseline_metrics = evaluate_models(baseline_models, X_train, X_test, y_train, y_test)
    
    if not baseline_metrics.empty:
        st.write("📊 Hasil Baseline Models:")
        st.dataframe(baseline_metrics.round(4))
    
    # === Hyperparameter Tuning ===
    st.subheader("🔧 Hyperparameter Tuning")
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
    
    tuned_models = {}
    best_params = {}
    
    progress_bar = st.progress(0)
    total_models = len(baseline_models)
    
    for i, (name, mdl) in enumerate(baseline_models.items()):
        try:
            st.text(f"🔄 Tuning model: {name}")
            
            if name == 'Random Forest':
                search = RandomizedSearchCV(
                    mdl, param_grids[name], n_iter=8, cv=3, 
                    scoring='f1_weighted', n_jobs=-1, random_state=42
                )
            else:
                search = GridSearchCV(
                    mdl, param_grids[name], cv=3, 
                    scoring='f1_weighted', n_jobs=-1
                )
            
            search.fit(X_train, y_train)
            tuned_models[name] = search.best_estimator_
            best_params[name] = search.best_params_
            
            st.success(f"✅ {name} - Best F1 Score: {search.best_score_:.4f}")
            with st.expander(f"Best Parameters - {name}"):
                st.json(search.best_params_)
                
        except Exception as e:
            st.error(f"❌ Error saat tuning {name}: {str(e)}")
            continue
        
        progress_bar.progress((i + 1) / total_models)
    
    # === Evaluation after tuning ===
    if tuned_models:
        st.subheader("📈 Evaluasi Model Setelah Tuning")
        
        with st.spinner("Evaluasi tuned models..."):
            tuned_metrics = evaluate_models(tuned_models, X_train, X_test, y_train, y_test, label='Tuned')
        
        if not tuned_metrics.empty:
            st.write("📊 Hasil Tuned Models:")
            st.dataframe(tuned_metrics.round(4))
        
        # === Visualization Comparison ===
        if not baseline_metrics.empty and not tuned_metrics.empty:
            st.subheader("📊 Perbandingan Performa Model")
            
            try:
                baseline_metrics_copy = baseline_metrics.copy()
                tuned_metrics_copy = tuned_metrics.copy()
                
                baseline_metrics_copy['Tipe'] = 'Baseline'
                tuned_metrics_copy['Tipe'] = 'Tuned'
                
                combined = pd.concat([baseline_metrics_copy, tuned_metrics_copy], ignore_index=True)
                melted = combined.melt(
                    id_vars=["Model", "Tipe"], 
                    value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                    var_name="Metric", 
                    value_name="Score"
                )
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=melted, x="Model", y="Score", hue="Tipe", ax=ax)
                plt.title("Perbandingan Performa Model – Baseline vs Tuned")
                plt.ylim(0, 1.05)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"❌ Error saat membuat visualisasi: {str(e)}")
        
        # === Conclusion ===
        st.subheader("🎯 Kesimpulan")
        
        try:
            if not baseline_metrics.empty and not tuned_metrics.empty:
                best_base = baseline_metrics.loc[baseline_metrics["F1 Score"].idxmax()]
                best_tuned = tuned_metrics.loc[tuned_metrics["F1 Score"].idxmax()]
                improvement = best_tuned["F1 Score"] - best_base["F1 Score"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Model Terbaik (Baseline)", 
                        best_base['Model'],
                        f"F1: {best_base['F1 Score']:.4f}"
                    )
                with col2:
                    st.metric(
                        "Model Terbaik (Tuned)", 
                        best_tuned['Model'],
                        f"F1: {best_tuned['F1 Score']:.4f}"
                    )
                
                st.metric("Peningkatan F1 Score", f"{improvement:.4f}")
                
                if improvement > 0.01:
                    st.success("🎉 Hyperparameter tuning berhasil meningkatkan performa model secara signifikan!")
                elif improvement > 0:
                    st.info("✅ Hyperparameter tuning sedikit meningkatkan performa model.")
                else:
                    st.warning("⚠️ Hyperparameter tuning tidak meningkatkan performa secara signifikan.")
            
        except Exception as e:
            st.error(f"❌ Error saat menghitung kesimpulan: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
