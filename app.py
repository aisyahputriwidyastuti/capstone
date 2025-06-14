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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Obesitas - Analisis Lengkap", layout="wide")
st.title("🏥 UAS Capstone Bengkel Koding - Klasifikasi Obesitas")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar untuk navigasi antar bagian
st.sidebar.title("📋 Navigasi")
sections = [
    "📊 EDA (Exploratory Data Analysis)",
    "🔄 Preprocessing Data", 
    "⚖️ SMOTE & Balancing",
    "🚀 Baseline Models",
    "🔧 Hyperparameter Tuning",
    "📈 Model Evaluation",
    "📊 Final Comparison",
    "🎯 Kesimpulan"
]

selected_section = st.sidebar.selectbox("Pilih Bagian:", sections)

@st.cache_data
def load_data():
    """Fungsi untuk memuat dataset dengan penanganan error"""
    try:
        df = pd.read_csv("ObesityDataSet.csv")
        st.success("✅ Dataset berhasil dimuat!")
        return df
    except FileNotFoundError:
        st.error("❌ File 'ObesityDataSet.csv' tidak ditemukan!")
        st.info("💡 Silakan upload file dataset atau pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"❌ Error saat memuat data: {str(e)}")
        return None

def comprehensive_eda(df):
    """Fungsi untuk melakukan Exploratory Data Analysis secara komprehensif"""
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    try:
        # Ringkasan dataset
        st.markdown("### 🔍 Ringkasan Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Baris", df.shape[0])
        with col2:
            st.metric("Total Kolom", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Duplikat", df.duplicated().sum())
        
        # Informasi detail dataset
        st.markdown("### 📋 Informasi Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**10 Baris Pertama:**")
            st.dataframe(df.head(10))
        
        with col2:
            st.markdown("**Info Dataset:**")
            # Membuat tabel informasi kolom
            info_data = []
            for col in df.columns:
                info_data.append({
                    'Kolom': col,
                    'Tipe': str(df[col].dtype),
                    'Non-Null': df[col].count(),
                    'Unique': df[col].nunique()
                })
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df)
        
        # Ringkasan statistik untuk kolom numerik
        st.markdown("### 📈 Ringkasan Statistik")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        
        # Analisis variabel target
        st.markdown("### 🎯 Analisis Variabel Target")
        if 'NObeyesdad' in df.columns:
            target_counts = df['NObeyesdad'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                # Grafik bar distribusi target
                fig, ax = plt.subplots(figsize=(10, 6))
                target_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Distribusi Kategori Obesitas')
                ax.set_xlabel('Kategori Obesitas')
                ax.set_ylabel('Jumlah')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Pie chart proporsi target
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
                ax.set_title('Proporsi Kategori Obesitas')
                st.pyplot(fig)
                plt.close()
            
            # Tabel distribusi target
            st.write("**Distribusi Target:**")
            target_df = pd.DataFrame({
                'Kategori': target_counts.index,
                'Jumlah': target_counts.values,
                'Persentase': (target_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(target_df)
        
        # Analisis fitur numerik
        st.markdown("### 📊 Analisis Fitur Numerik")
        if len(numeric_cols) > 0:
            # Plot distribusi untuk setiap fitur numerik
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    ax = axes[i] if len(numeric_cols) > 1 else axes[0]
                    df[col].hist(bins=30, ax=ax, alpha=0.7)
                    ax.set_title(f'Distribusi {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frekuensi')
            
            # Sembunyikan subplot kosong
            for j in range(len(numeric_cols), len(axes)):
                if j < len(axes):
                    axes[j].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Matriks korelasi
            st.markdown("### 🔗 Matriks Korelasi")
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matriks Korelasi Fitur Numerik')
            st.pyplot(fig)
            plt.close()
        
        # Analisis fitur kategorikal
        st.markdown("### 📊 Analisis Fitur Kategorikal")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols[:6]:  # Tampilkan 6 kolom kategorikal pertama
                if df[col].nunique() < 20:  # Hanya tampilkan jika tidak terlalu banyak kategori
                    fig, ax = plt.subplots(figsize=(10, 4))
                    value_counts = df[col].value_counts()
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Distribusi {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Jumlah')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        # Deteksi outlier
        st.markdown("### 🔍 Deteksi Outlier")
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(4, len(numeric_cols)), figsize=(20, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            elif len(numeric_cols) < 4:
                axes = list(axes[:len(numeric_cols)])
            
            for i, col in enumerate(numeric_cols[:4]):
                ax = axes[i] if len(numeric_cols) > 1 else axes[0]
                df.boxplot(column=col, ax=ax)
                ax.set_title(f'Boxplot {col}')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"❌ Error dalam EDA: {str(e)}")

def remove_outliers_iqr(data, column):
    """Fungsi untuk menghapus outlier menggunakan metode IQR"""
    try:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]
    except Exception as e:
        st.error(f"Error removing outliers for {column}: {str(e)}")
        return data

def preprocessing_analysis(df):
    """Fungsi untuk melakukan analisis preprocessing secara komprehensif"""
    st.subheader("🔄 Preprocessing Data")
    
    try:
        # Informasi data asli
        st.markdown("### 📋 Data Asli")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baris", df.shape[0])
        with col2:
            st.metric("Kolom", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Langkah 1: Menghapus duplikat dan missing values
        st.markdown("### 🧹 Langkah 1: Membersihkan Data")
        original_rows = len(df)
        df_clean = df.copy()
        df_clean.drop_duplicates(inplace=True)
        df_clean.dropna(inplace=True)
        
        rows_removed = original_rows - len(df_clean)
        st.write(f"Baris yang dihapus (duplikat/missing): {rows_removed}")
        
        # Langkah 2: Menghapus outlier
        st.markdown("### 🎯 Langkah 2: Menghapus Outlier")
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        
        outlier_info = []
        df_no_outliers = df_clean.copy()
        
        for col in numeric_cols:
            before = len(df_no_outliers)
            df_no_outliers = remove_outliers_iqr(df_no_outliers, col)
            after = len(df_no_outliers)
            outliers_removed = before - after
            outlier_info.append({
                'Kolom': col,
                'Outlier Dihapus': outliers_removed,
                'Baris Tersisa': after
            })
        
        if outlier_info:
            outlier_df = pd.DataFrame(outlier_info)
            st.dataframe(outlier_df)
        
        total_outliers = len(df_clean) - len(df_no_outliers)
        st.write(f"**Total baris yang dihapus (outlier): {total_outliers}**")
        
        # Langkah 3: Encoding variabel kategorikal
        st.markdown("### 🔄 Langkah 3: Encoding Variabel Kategorikal")
        
        # Identifikasi kolom kategorikal
        cat_cols = df_no_outliers.select_dtypes(include='object').columns.tolist()
        if 'NObeyesdad' in cat_cols:
            cat_cols.remove('NObeyesdad')  # Hapus target dari daftar fitur
        
        # Simpan encoder untuk setiap kolom
        label_encoders = {}
        df_encoded = df_no_outliers.copy()
        
        encoding_info = []
        for col in cat_cols:
            le = LabelEncoder()
            original_values = df_encoded[col].unique()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
            
            encoding_info.append({
                'Kolom': col,
                'Kategori Asli': len(original_values),
                'Contoh Mapping': f"{original_values[0]} -> {le.transform([original_values[0]])[0]}"
            })
        
        if encoding_info:
            encoding_df = pd.DataFrame(encoding_info)
            st.dataframe(encoding_df)
        
        # Encoding target variable
        target_encoder = None
        if 'NObeyesdad' in df_encoded.columns:
            target_encoder = LabelEncoder()
            original_target = df_encoded['NObeyesdad'].unique()
            df_encoded['NObeyesdad'] = target_encoder.fit_transform(df_encoded['NObeyesdad'])
            
            st.write("**Encoding Target Variable:**")
            target_mapping = {orig: int(encoded) for orig, encoded in zip(original_target, 
                             target_encoder.transform(original_target))}
            st.json(target_mapping)
        
        # Ringkasan preprocessing
        st.markdown("### 📊 Ringkasan Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Asli", original_rows)
        with col2:
            st.metric("Data Bersih", len(df_clean))
        with col3:
            st.metric("Data Final", len(df_encoded))
        
        return df_encoded, label_encoders, target_encoder
        
    except Exception as e:
        st.error(f"❌ Error dalam preprocessing: {str(e)}")
        return df, {}, None

def smote_analysis(df, target_col='NObeyesdad'):
    """Fungsi untuk melakukan analisis SMOTE dan balancing"""
    st.subheader("⚖️ SMOTE & Balancing")
    
    try:
        # Persiapan data
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Distribusi kelas sebelum SMOTE
        st.markdown("### 📊 Distribusi Kelas Sebelum SMOTE")
        before_smote = y.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah sampel per kelas:**")
            st.dataframe(before_smote.to_frame('Jumlah'))
        
        with col2:
            # Visualisasi distribusi sebelum SMOTE
            fig, ax = plt.subplots(figsize=(8, 6))
            before_smote.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Distribusi Kelas Sebelum SMOTE')
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Jumlah Sampel')
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()
        
        # Aplikasi SMOTE
        st.markdown("### 🔄 Aplikasi SMOTE")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Distribusi kelas setelah SMOTE
        after_smote = pd.Series(y_res).value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah sampel per kelas setelah SMOTE:**")
            st.dataframe(after_smote.to_frame('Jumlah'))
        
        with col2:
            # Visualisasi distribusi setelah SMOTE
            fig, ax = plt.subplots(figsize=(8, 6))
            after_smote.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Distribusi Kelas Setelah SMOTE')
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Jumlah Sampel')
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()
        
        # Perbandingan sebelum dan sesudah
        st.markdown("### 📊 Perbandingan Sebelum dan Sesudah SMOTE")
        comparison_df = pd.DataFrame({
            'Kelas': before_smote.index,
            'Sebelum SMOTE': before_smote.values,
            'Sesudah SMOTE': after_smote.values,
            'Peningkatan': after_smote.values - before_smote.values
        })
        st.dataframe(comparison_df)
        
        # Grafik perbandingan
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['Sebelum SMOTE'], width, 
               label='Sebelum SMOTE', color='lightcoral')
        ax.bar(x + width/2, comparison_df['Sesudah SMOTE'], width, 
               label='Sesudah SMOTE', color='lightgreen')
        
        ax.set_title('Perbandingan Distribusi Kelas')
        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah Sampel')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Kelas'])
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.success("✅ SMOTE berhasil diaplikasikan!")
        return X_res, y_res
        
    except Exception as e:
        st.error(f"❌ Error saat aplikasi SMOTE: {str(e)}")
        return X, y

def standardization_analysis(X, feature_names):
    """Fungsi untuk melakukan analisis standardisasi"""
    try:
        st.markdown("### 📏 Standardisasi Fitur")
        
        # Konversi ke DataFrame jika diperlukan
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Statistik sebelum standardisasi
        st.write("**Statistik Sebelum Standardisasi:**")
        before_stats = pd.DataFrame({
            'Mean': X.mean(),
            'Std': X.std(),
            'Min': X.min(),
            'Max': X.max()
        })
        st.dataframe(before_stats.round(4))
        
        # Aplikasi standardisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Statistik setelah standardisasi
        st.write("**Statistik Setelah Standardisasi:**")
        after_stats = pd.DataFrame({
            'Mean': X_scaled.mean(),
            'Std': X_scaled.std(),
            'Min': X_scaled.min(),
            'Max': X_scaled.max()
        })
        st.dataframe(after_stats.round(4))
        
        # Visualisasi perbandingan (hanya 5 fitur pertama)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sebelum standardisasi
        X.iloc[:, :5].boxplot(ax=ax1)
        ax1.set_title('Distribusi Sebelum Standardisasi')
        ax1.set_ylabel('Nilai')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Setelah standardisasi
        X_scaled.iloc[:, :5].boxplot(ax=ax2)
        ax2.set_title('Distribusi Setelah Standardisasi')
        ax2.set_ylabel('Nilai')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        return X_scaled, scaler
        
    except Exception as e:
        st.error(f"❌ Error dalam standardisasi: {str(e)}")
        return X, None

def evaluate_models_comprehensive(models_dict, X_train, X_test, y_train, y_test, label='Model'):
    """Fungsi untuk evaluasi model secara komprehensif"""
    results = {
        'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 
        'F1 Score': [], 'ROC AUC': [], 'CV Score Mean': [], 'CV Score Std': []
    }
    
    for name, model in models_dict.items():
        try:
            st.write(f"🔄 Mengevaluasi {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Hitung metrik
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighte
