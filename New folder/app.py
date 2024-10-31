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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
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

def  clean_numeric_columns(data):
    """Remove everything except numeric characters (0-9) from object-type columns."""
    for column in data.select_dtypes(include=['object']):
        # Keep only digits (0-9)
        data[column] = data[column].str.replace(r'[^0-9]', '', regex=True)

    return data

#def clean_numeric_columns(data):
    """Clean all columns in the DataFrame that can be converted to numeric."""
    for column in data.select_dtypes(include=['object']):  # Iterate over object (string) columns
        # Clean the column
        cleaned_column = (
            data[column]
            .str.replace('K', '', regex=False)  # Remove 'K'
            .str.replace('−', '-', regex=False)  # Replace special minus
            .str.replace(' ', '', regex=False)  # Remove spaces
        )
        # Convert to numeric, coercing errors to NaN
        data[column] = pd.to_numeric(cleaned_column, errors='coerce')
    return data

# Function to create downloadable text file
def create_downloadable_report(report):
    buffer = io.StringIO()
    buffer.write(report)
    buffer.seek(0)
    return buffer

# Streamlit app title
st.title("📊 Advanced CSV Data Insights and ML App")
st.markdown("Upload your CSV file to analyze data, visualize insights, and apply machine learning.")

# Create a sidebar for file upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state variables
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None

data = None

if uploaded_file is not None:
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
    else:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file uploaded successfully!")

            if data.empty:
                st.error("The uploaded CSV file does not contain any data. Please upload a valid CSV file.")
            else:
                # Display original shape of the dataset
                original_shape = data.shape
                st.write(f"Original dataset shape: {original_shape}")
                
                # Data Cleaning Options
                st.sidebar.subheader("🧹 Data Cleaning Options")
                missing_value_method = st.sidebar.selectbox(
                    "Choose missing value handling method:", 
                    ["None", "Clean Numeric column", "Drop Rows", "Simple Imputer (Mean)", "KNN Imputer"]
                )
                if missing_value_method=="Clean Numeric column":
                    data_before=data.shape      
                    # Clean all numeric columns
                    data = clean_numeric_columns(data)
                    st.success("Cleaned all numeric columns.")


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

                if st.sidebar.checkbox("Drop duplicate rows"):
                    data_before = data.shape
                    data = data.drop_duplicates()
                    st.sidebar.success("Duplicate rows have been dropped.")
                    st.write(f"Shape after dropping duplicates: {data.shape} (Removed {data_before[0] - data.shape[0]} rows)")

                # New Preprocessing Techniques
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

                # Data Type Conversion Option
                st.sidebar.subheader("🔄 Data Type Conversion")
                for column in data.columns:
                    new_dtype = st.sidebar.selectbox(f"Convert {column} to:", options=["No change", "Integer", "Float", "Category", "Object"], index=0)
                    if new_dtype == "Integer":
                        data[column] = data[column].astype(int)
                    elif new_dtype == "Float":
                        data[column] = data[column].astype(float)
                    elif new_dtype == "Category":
                        data[column] = data[column].astype('category')
                    elif new_dtype == "Object":
                        data[column] = data[column].astype(str)

                # Display DataFrame preview in the center
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
                    plt.figure(figsize=(10, 5))
                    
                    if plot_type == "Histogram":
                        st.subheader(f"Histogram of {column}")
                        sns.histplot(data[column], bins=30, kde=True)
                        plt.xlabel(column)
                        plt.ylabel("Frequency")
                        st.pyplot(plt)

                    elif plot_type == "Box Plot":
                        st.subheader(f"Box Plot of {column}")
                        sns.boxplot(y=data[column])
                        plt.ylabel(column)
                        st.pyplot(plt)

                    elif plot_type == "Scatter Plot":
                        numeric_cols = data.select_dtypes(include=['float64', 'int']).columns.tolist()
                        if len(numeric_cols) >= 2:
                            x_column = numeric_cols[0]
                            y_column = numeric_cols[1]
                            st.subheader(f"Scatter Plot of {y_column} vs {x_column}")
                            sns.scatterplot(data=data, x=x_column, y=y_column)
                            plt.xlabel(x_column)
                            plt.ylabel(y_column)
                            st.pyplot(plt)
                        else:
                            st.warning("Not enough numeric columns for scatter plot.")

                    elif plot_type == "Pair Plot":
                        st.subheader("Pair Plot")
                        sns.pairplot(data)
                        st.pyplot(plt)

                    elif plot_type == "Heatmap":
                        st.subheader("Heatmap of Correlations")
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(data.corr(), annot=True, fmt=".2f")
                        plt.title("Correlation Heatmap")
                        st.pyplot(plt)

                    elif plot_type == "Line Plot":
                        if len(data.columns) >= 2:
                            x_column = data.columns[0]
                            y_column = data.columns[1]
                            st.subheader(f"Line Plot of {y_column} vs {x_column}")
                            plt.plot(data[x_column], data[y_column])
                            plt.xlabel(x_column)
                            plt.ylabel(y_column)
                            st.pyplot(plt)
                        else:
                            st.warning("Not enough columns for line plot.")

                    elif plot_type == "Bar Plot":
                        st.subheader(f"Bar Plot of {column}")
                        sns.barplot(x=data.index, y=data[column])
                        plt.xlabel("Index")
                        plt.ylabel(column)
                        st.pyplot(plt)

                    elif plot_type == "Pie Chart":
                        st.subheader(f"Pie Chart of {column}")
                        plt.pie(data[column].value_counts(), labels=data[column].value_counts().index, autopct='%1.1f%%')
                        st.pyplot(plt)

                    elif plot_type == "Area Chart":
                        st.subheader(f"Area Chart of {column}")
                        plt.fill_between(data.index, data[column])
                        plt.xlabel("Index")
                        plt.ylabel(column)
                        st.pyplot(plt)

                # Display basic data insights
                st.subheader("📋 Basic Data Insights")
                insights = generate_basic_insights(data)
                st.json(insights)

                # Feature Engineering: Feature Selection
                st.sidebar.subheader("📊 Feature Selection")
                select_k_best = st.sidebar.slider("Number of Top Features to Select", 1, min(10, data.shape[1]), 5)
                if st.sidebar.button("Run Feature Selection"):
                    selector = SelectKBest(score_func=f_classif, k=select_k_best)
                    X_selected = selector.fit_transform(data.iloc[:, :-1], data.iloc[:, -1])
                    selected_features = data.columns[selector.get_support()]
                    st.sidebar.success(f"Top {select_k_best} features selected.")
                    st.write(f"Selected Features: {selected_features}")

                # Machine Learning Model Selection
                st.sidebar.subheader("🤖 Machine Learning")
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                model_choice = st.sidebar.selectbox("Select ML Model:", 
                                                    ["Logistic Regression", "Decision Tree", "Random Forest", 
                                                     "Gradient Boosting", "SVM", "KNN", "XGBoost"])
                scale_option = st.sidebar.selectbox("Choose Scaling Method:", 
                                                    ["None", "Standard Scaler", "Min-Max Scaler", "Robust Scaler", "Normalizer"])

                if scale_option == "Standard Scaler":
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                elif scale_option == "Min-Max Scaler":
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)
                elif scale_option == "Robust Scaler":
                    scaler = RobustScaler()
                    X = scaler.fit_transform(X)
                elif scale_option == "Normalizer":
                    scaler = Normalizer()
                    X = scaler.fit_transform(X)

                # Splitting the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if st.sidebar.button("Train Model"):
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression()
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeClassifier()
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier()
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingClassifier()
                    elif model_choice == "SVM":
                        model = SVC()
                    elif model_choice == "KNN":
                        model = KNeighborsClassifier()
                    elif model_choice == "XGBoost":
                        model = XGBClassifier()

                    # Hyperparameter Tuning with Optuna
                    def objective(trial):
                        if model_choice == "Random Forest":
                            n_estimators = trial.suggest_int("n_estimators", 50, 300)
                            max_depth = trial.suggest_int("max_depth", 2, 10)
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                        elif model_choice == "SVM":
                            C = trial.suggest_loguniform('C', 1e-3, 1e2)
                            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
                            model = SVC(C=C, kernel=kernel)
                        elif model_choice == "Gradient Boosting":
                            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
                            n_estimators = trial.suggest_int("n_estimators", 50, 300)
                            model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
                        elif model_choice == "XGBoost":
                            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.1)
                            n_estimators = trial.suggest_int("n_estimators", 50, 300)
                            model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
                        else:
                            model = model

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        accuracy = accuracy_score(y_test, preds)
                        return accuracy

                    st.write("Optimizing model...")
                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=10)
                    best_params = study.best_params
                    st.write(f"Best Parameters: {best_params}")

                    # Train the model with best parameters
                    model.set_params(**best_params)
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.sidebar.success("Model trained successfully!")

                # Predict and Evaluate Model
                if st.session_state.model_trained:
                    preds = st.session_state.model.predict(X_test)
                    st.write("Model Evaluation Metrics:")
                    st.write("Accuracy:", accuracy_score(y_test, preds))
                    st.write("ROC AUC:", roc_auc_score(y_test, preds, multi_class='ovr'))
                    st.write("Confusion Matrix:", confusion_matrix(y_test, preds))
                    st.write("Classification Report:", classification_report(y_test, preds))

                    # Model Explainability
                    st.subheader("🧠 Model Explainability")
                    explain_option = st.selectbox("Choose explainability method:", ["SHAP", "LIME"])
                    if explain_option == "SHAP":
                        st.write("SHAP plots are not yet integrated in this app.")
                    elif explain_option == "LIME":
                        st.write("LIME explanation coming soon!")

                # Generate Summary Report
                st.subheader("📄 Summary Report")
                report_agent = create_report_writer_agent()
                report = generate_report(report_agent, insights)
                st.write(report)

                # Provide option to download the report
                st.download_button(
                    label="Download Report",
                    data=create_downloadable_report(report),
                    file_name="summary_report.txt",
                    mime="text/plain"
                )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError:
            st.error("The uploaded CSV file could not be parsed. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please upload a CSV file to start the analysis.")
