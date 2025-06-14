import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

# Config
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("Aplikasi Prediksi Obesitas")

# Sidebar
menu = st.sidebar.selectbox("Menu", ["EDA", "Preprocessing", "Modeling", "Hyperparameter Tuning", "Prediksi"])

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("ObesityDataSet.csv")
    except:
        # Sample data jika file tidak ada
        np.random.seed(42)
        data = {
            'Gender': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.normal(25, 5, 200),
            'Height': np.random.normal(1.7, 0.1, 200),
            'Weight': np.random.normal(70, 15, 200),
            'FAVC': np.random.choice(['yes', 'no'], 200),
            'FCVC': np.random.randint(1, 4, 200),
            'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently'], 200),
            'SMOKE': np.random.choice(['yes', 'no'], 200),
            'CH2O': np.random.normal(2, 0.5, 200),
            'FAF': np.random.normal(1, 0.5, 200),
            'NObeyesdad': np.random.choice(['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I'], 200)
        }
        return pd.DataFrame(data)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}

# EDA
if menu == "EDA":
    st.header("Exploratory Data Analysis")
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Target Distribution")
    if 'NObeyesdad' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['NObeyesdad'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribusi Kategori Obesitas')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Preprocessing
elif menu == "Preprocessing":
    st.header("Data Preprocessing")
    df = load_data()
    
    # Clean data
    df_clean = df.dropna().drop_duplicates()
    st.write(f"Data setelah cleaning: {df_clean.shape}")
    
    # Encode categorical variables
    df_encoded = df_clean.copy()
    encoders = {}
    
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col != 'NObeyesdad':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    # Encode target
    le_target = LabelEncoder()
    df_encoded['target'] = le_target.fit_transform(df_encoded['NObeyesdad'])
    df_encoded.drop('NObeyesdad', axis=1, inplace=True)
    encoders['target'] = le_target
    
    # Features and target
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Store processed data
    st.session_state.processed_data = {
        'X': X_scaled,
        'y': y_balanced,
        'encoders': encoders,
        'scaler': scaler
    }
    
    st.success("Preprocessing selesai!")
    st.write(f"Data akhir: {X_scaled.shape}")

# Modeling
elif menu == "Modeling":
    st.header("Modeling dan Evaluasi")
    
    if st.session_state.processed_data is None:
        st.error("Silakan jalankan preprocessing terlebih dahulu!")
    else:
        data = st.session_state.processed_data
        X = data['X']
        y = data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier()
        }
        
        results = {}
        
        # Train models
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        
        st.session_state.models = results
        
        # Display results
        result_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[k]['accuracy'] for k in results.keys()],
            'F1-Score': [results[k]['f1_score'] for k in results.keys()]
        })
        
        st.dataframe(result_df)
        
        # Best model
        best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
        st.success(f"Model terbaik: {best_model}")

# Hyperparameter Tuning
elif menu == "Hyperparameter Tuning":
    st.header("Hyperparameter Tuning")
    
    if not st.session_state.models:
        st.error("Silakan jalankan modeling terlebih dahulu!")
    else:
        data = st.session_state.processed_data
        X = data['X']
        y = data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Grid search for Random Forest
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        st.write("Best parameters:", grid_search.best_params_)
        st.write("Best score:", f"{grid_search.best_score_:.4f}")
        
        # Store best model
        st.session_state.best_model = grid_search.best_estimator_

# Prediksi
elif menu == "Prediksi":
    st.header("Prediksi Obesitas")
    
    if 'best_model' not in st.session_state:
        st.error("Silakan selesaikan semua tahap terlebih dahulu!")
    else:
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        
        with col2:
            favc = st.selectbox("High Caloric Food", ["yes", "no"])
            fcvc = st.selectbox("Vegetable Frequency", [1, 2, 3])
            caec = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently"])
            smoke = st.selectbox("Smoking", ["yes", "no"])
        
        ch2o = st.number_input("Water Consumption (L/day)", min_value=0.5, max_value=5.0, value=2.0)
        faf = st.number_input("Physical Activity", min_value=0.0, max_value=3.0, value=1.0)
        
        if st.button("Prediksi"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Height': [height],
                'Weight': [weight],
                'FAVC': [favc],
                'FCVC': [fcvc],
                'CAEC': [caec],
                'SMOKE': [smoke],
                'CH2O': [ch2o],
                'FAF': [faf]
            })
            
            # Encode and predict
            try:
                data = st.session_state.processed_data
                encoders = data['encoders']
                scaler = data['scaler']
                
                # Simple encoding (you may need to adjust based on your actual encoders)
                for col in input_data.select_dtypes(include=['object']).columns:
                    input_data[col] = input_data[col].map({'yes': 1, 'no': 0, 'Male': 1, 'Female': 0, 
                                                          'Sometimes': 1, 'Frequently': 2}).fillna(0)
                
                # Scale
                input_scaled = scaler.transform(input_data)
                
                # Predict
                prediction = st.session_state.best_model.predict(input_scaled)[0]
                
                # Decode prediction
                obesity_levels = ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I']
                result = obesity_levels[prediction] if prediction < len(obesity_levels) else 'Unknown'
                
                st.success(f"Prediksi Tingkat Obesitas: {result}")
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
