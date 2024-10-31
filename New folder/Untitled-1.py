import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from langchain_ollama import OllamaLLM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier
import optuna

# Function to generate basic insights from the DataFrame
def generate_basic_insights(data):
    insights = {
        'columns': data.columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict(),
        'description': data.describe(include='all').to_dict()
    }
    return insights

# Function to create the report writer agent
def create_report_writer_agent():
    llm = OllamaLLM(model="llama3.1")
    return llm

# Function to generate a summary report
def generate_report(llm, insights):
    prompt = f"""
    Based on the following insights, create a concise summary report:
    Insights:
    {insights}
    """
    report = llm.invoke(prompt)
    return report

# Function to create downloadable text file
def create_downloadable_report(report):
    buffer = io.StringIO()
    buffer.write(report)
    buffer.seek(0)
    return buffer

# Streamlit app title
st.title("📊 Advanced CSV Data Insights and ML App")
st.markdown("Upload your CSV file to analyze data, visualize insights, and apply machine learning.")

# Sidebar for file upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state variables
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None

data = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV file uploaded successfully!")

        if data.empty:
            st.error("The uploaded CSV file does not contain any data. Please upload a valid CSV file.")
        else:
            st.write(f"Original dataset shape: {data.shape}")

            # Data Cleaning Options
            st.sidebar.subheader("🧹 Data Cleaning Options")
            missing_value_method = st.sidebar.selectbox(
                "Choose missing value handling method:", 
                ["None", "Drop Rows", "Simple Imputer (Mean)", "KNN Imputer"]
            )

            # Handle missing values
            if missing_value_method == "Drop Rows":
                data_before = data.shape
                data = data.dropna()
                st.sidebar.success(f"Dropped rows with missing values. New shape: {data.shape}")
                
            elif missing_value_method == "Simple Imputer (Mean)":
                imputer = SimpleImputer(strategy='mean')
                data[:] = imputer.fit_transform(data)
                st.sidebar.success("Applied Simple Imputer (Mean).")
                
            elif missing_value_method == "KNN Imputer":
                imputer = KNNImputer(n_neighbors=3)
                data[:] = imputer.fit_transform(data)
                st.sidebar.success("Applied KNN Imputer.")

            # Outlier Detection
            st.sidebar.subheader("🔍 Outlier Detection")
            if st.sidebar.checkbox("Remove Outliers (Z-Score Method)"):
                z_scores = np.abs((data - data.mean()) / data.std())
                data = data[(z_scores < 3).all(axis=1)]
                st.sidebar.success("Removed outliers using Z-Score method.")

            # Drop duplicate rows
            if st.sidebar.checkbox("Drop duplicate rows"):
                data_before = data.shape
                data = data.drop_duplicates()
                st.sidebar.success("Duplicate rows have been dropped.")
                st.write(f"Shape after dropping duplicates: {data.shape} (Removed {data_before[0] - data.shape[0]} rows)")

            # Additional Preprocessing Options
            st.sidebar.subheader("🔧 Additional Preprocessing Options")
            if st.sidebar.checkbox("Label Encode Categorical Columns"):
                label_encoders = {}
                for column in data.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column])
                    label_encoders[column] = le
                st.sidebar.success("Categorical columns have been label encoded.")
                
            if st.sidebar.checkbox("One-Hot Encode Categorical Columns"):
                data = pd.get_dummies(data)
                st.sidebar.success("Categorical columns have been one-hot encoded.")
                
            if st.sidebar.checkbox("Apply PCA"):
                n_components = st.sidebar.slider("Select number of components for PCA:", 1, min(data.shape[1], 10), 2)
                pca = PCA(n_components=n_components)
                data_pca = pca.fit_transform(data)
                data = pd.DataFrame(data_pca, columns=[f"PC{i+1}" for i in range(n_components)])
                st.sidebar.success("PCA has been applied.")
                st.write(f"Shape after PCA: {data.shape}")

            # Display DataFrame preview
            st.subheader("👀 Data Preview:")
            st.write(data.head())

            # Show basic statistics
            st.subheader("📊 Basic Statistics:")
            st.write(data.describe())

            # Graphical Insights Section
            st.subheader("📊 Graphical Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Select a column for graphical insights:**")
                column = st.selectbox("Choose a column:", data.columns)

            with col2:
                st.markdown("**Select plot type:**")
                plot_type = st.selectbox("Select plot type:", 
                                          ["Histogram", "Box Plot", "Scatter Plot", 
                                           "Pair Plot", "Heatmap", "Line Plot", "Bar Plot", "Pie Chart", "Area Chart"])

            # Generate visualizations based on user selection
            if st.button("Generate Graph"):
                if plot_type == "Histogram":
                    st.subheader(f"Histogram of {column}")
                    fig = px.histogram(data, x=column, nbins=30, title=f'Histogram of {column}')
                    st.plotly_chart(fig)

                elif plot_type == "Box Plot":
                    st.subheader(f"Box Plot of {column}")
                    fig = px.box(data, y=column, title=f'Box Plot of {column}')
                    st.plotly_chart(fig)

                elif plot_type == "Scatter Plot":
                    numeric_cols = data.select_dtypes(include=['float64', 'int']).columns.tolist()
                    if len(numeric_cols) >= 2:
                        x_column = st.selectbox("Select X column:", numeric_cols)
                        y_column = st.selectbox("Select Y column:", numeric_cols, index=1)
                        st.subheader(f"Scatter Plot of {y_column} vs {x_column}")
                        fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter Plot of {y_column} vs {x_column}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("Not enough numeric columns for scatter plot.")

                elif plot_type == "Pair Plot":
                    st.subheader("Pair Plot")
                    fig = px.scatter_matrix(data)
                    st.plotly_chart(fig)

                elif plot_type == "Heatmap":
                    st.subheader("Heatmap of Correlations")
                    fig = px.imshow(data.corr(), text_auto=True, title="Correlation Heatmap")
                    st.plotly_chart(fig)

                elif plot_type == "Line Plot":
                    if len(data.columns) >= 2:
                        x_column = data.columns[0]
                        y_column = data.columns[1]
                        st.subheader(f"Line Plot of {y_column} vs {x_column}")
                        fig = px.line(data, x=x_column, y=y_column, title=f'Line Plot of {y_column} vs {x_column}')
                        st.plotly_chart(fig)
                    else:
                        st.warning("Not enough columns for line plot.")

                elif plot_type == "Bar Plot":
                    st.subheader(f"Bar Plot of {column}")
                    fig = px.bar(data, x=data.index, y=column, title=f'Bar Plot of {column}')
                    st.plotly_chart(fig)

                elif plot_type == "Pie Chart":
                    st.subheader(f"Pie Chart of {column}")
                    fig = px.pie(data, names=column, title=f'Pie Chart of {column}')
                    st.plotly_chart(fig)

                elif plot_type == "Area Chart":
                    st.subheader(f"Area Chart of {column}")
                    fig = px.area(data, x=data.index, y=column, title=f'Area Chart of {column}')
                    st.plotly_chart(fig)

            # Display basic data insights
            st.subheader("📋 Basic Data Insights")
            insights = generate_basic_insights(data)
            st.json(insights)

            # Feature Engineering: Feature Selection
            st.sidebar
