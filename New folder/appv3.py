import streamlit as st
import pandas as pd
import io
from langchain_ollama import OllamaLLM

# Function to generate basic insights from the DataFrame
def generate_basic_insights(data):
    try:
        # Generate insights as DataFrames or Series
        insights = {
            'columns': pd.Series(data.columns.tolist(), name="Columns"),
            'missing_values': data.isnull().sum(),  # Series of missing value counts
            'data_types': data.dtypes,  # Series of data types
            'description': data.describe(include='all')  # DataFrame with descriptive stats
        }
        return insights
    except Exception as e:
        print(f"Error generating insights: {e}")
        return {}

# Function to create the report writer agent
def create_report_writer_agent():
    try:
        # Assuming OllamaLLM needs to be initialized like this
        llm = OllamaLLM(model="llama3.1")
        return llm
    except Exception as e:
        print(f"Error creating report writer agent: {e}")
        return None

# Function to generate a summary report
def generate_report(llm, insights):
    if not llm:
        return "Error: LLM not initialized."
    
    # Format the insights for the model in a readable way
    formatted_insights = (
        f"Columns in the DataFrame: {insights['columns'].tolist()}\n"
        f"Missing values per column:\n{insights['missing_values']}\n"
        f"Data types of columns:\n{insights['data_types']}\n"
        f"Statistical summary:\n{insights['description']}"
    )
    
    prompt = f"""
    Based on the following insights, create a concise summary report:
    {formatted_insights}
    """
    
    try:
        # Assuming the LLM API is invoked like this
        report = llm.invoke(prompt)
        return report
    except Exception as e:
        return f"Error generating report: {e}"

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

# Check if a file was uploaded
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Generate insights for the uploaded data
        insights = generate_basic_insights(data)

        # Display the basic insights
        st.subheader("📊 Basic Insights")
        st.write(f"Columns: {insights['columns'].tolist()}")
        st.write(f"Missing values: {insights['missing_values']}")
        st.write(f"Data Types: {insights['data_types']}")
        st.write(f"Statistical Summary: {insights['description']}")

        # Generate Summary Report
        st.subheader("📄 Summary Report")
        report_agent = create_report_writer_agent()
        if report_agent:
            report = generate_report(report_agent, insights)
            st.write(report)

            # Provide option to download the report
            st.download_button(
                label="Download Report",
                data=create_downloadable_report(report),
                file_name="summary_report.txt",
                mime="text/plain"
            )
        else:
            st.error("Failed to create report writer agent.")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
    except pd.errors.ParserError:
        st.error("The uploaded CSV file could not be parsed. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please upload a CSV file to start the analysis.")
