import time

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from sql_metadata import Parser
import mysql.connector
import os
from dotenv import load_dotenv
import plotly_express as px

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database connection
username = "root"
password = 'feb182001'
host = "localhost"
database = "sakila"

# create a connection to MYSQL server
engine = mysql.connector.connect(user=username, password=password, host=host, database=database)


# Gemini model for SQL query modification
gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)



# Tool to get the table schema
@st.cache_data
def extract_table_names(query):
    try:
        tables = Parser(query).tables
        return tables
    except:
        return "error in parsing sql query"
@st.cache_data
def get_table_schema(table):
    '''This function takes in table and output the SQL schema'''
    query = f'''show create table {table}'''
    table_schema = pd.read_sql_query(query, engine)
    schema = table_schema['Create Table'].iloc[0]
    index = schema.find("ENGINE")
    schema = schema[:index]
    return schema

@st.cache_data
def modify_sql_query(input_query, question,schema):
    """
    Modifies an SQL query based on a user's question using Gemini.

    Args:
        input_query (str): The original SQL query.
        question (str): The user's question related to the query.

    Returns:
        str: The modified SQL query, or None if an error occurs.
    """
    try:
        # Construct the prompt for Gemini
        prompt = f""" You are a powerful assistant specialized in modifying SQL queries for effective data visualization.

                    Instructions:
                    1. Modify the SQL query in a way that provides data for visualization.
                    2. Use the input query as the primary reference and the question as your secondary reference when modifying the query.
                    3. Use the table schema to get the column names and types when modifying the query.
                    4. Keep the original query intact as much as possible.
                    5. Always output only the modified SQL query without any additional text.
                    6. Do not make up your own SQL queries.
                    
                    INPUTS:
                    Input Query: {input_query}
                    Question: {question}
                    Table Schema: {schema}
                    Modified SQL Query: [Provide the modified SQL query here without triple backticks]
                    """

        # Use Gemini to modify the SQL query
        gemini_response = gemini_model.invoke(prompt)

        # Extract the modified SQL query from Gemini's response
        modified_sql_query = gemini_response.content.strip()

        # Replace BIGNUMERIC and INT64 with appropriate MySQL types
        modified_sql_query = modified_sql_query.replace("BIGNUMERIC", "DECIMAL(18,2)")
        modified_sql_query = modified_sql_query.replace("INT64", "INT")

        return modified_sql_query
    except Exception as e:
        print(f"Error modifying SQL query: {e}")
        return None

@st.cache_data
def get_data_frame(modified_sql_query):
    df = pd.read_sql_query(modified_sql_query, engine)
    return df

# data description
@st.cache_data
def data_description_modified(modified_sql_query):
    df = get_data_frame(modified_sql_query)
    columns = df.columns
    description = []
    for column in columns:
        col = []
        unique = df[column].unique()
        col_description = f"Column_Name: {column}, sample_data: {list(unique[:3])}, number_of_unique_values: {len(unique)}, dtype: {unique[0].__class__.__name__}"
        col.append(col_description)
        description.append(col)
    return description



# Define gemini_suggest_plot function


# Prompt template for gemini_suggest_plot
template_gemini_suggest_plot = """
You are a powerful assistant specialized in suggesting the best plot type based on the characteristics of the data.

Instructions:
1. Analyze the provided data description and suggest the most suitable plot type.
2. Consider the unique values, data types, etc., of each column when making your suggestion.
3. Output only the recommended plot type without any additional text.
4. Always suggest the best plot according to the data description

Inputs:
Data Description: {data_frame_preview}

Recommended Plot Type: 
"""

# Prompt template using format
@st.cache_data
def prompt_gemini_suggest_plot(data_frame_preview):
    return template_gemini_suggest_plot.format(
        data_frame_preview=data_frame_preview
    )
@st.cache_data
def gemini_suggest_plot(df):
    # Convert DataFrame information into a format that Gemini understands
    column_descriptions = []
    for column in df.columns:
        unique_values = df[column].unique()
        sample_values = list(unique_values[:3])
        n = len(df)
        column_description = f"Column_Name: {column}, sample_data: {sample_values}, number_of_rows: {n} dtype: {str(df[column].dtype)}"
        column_descriptions.append(column_description)

    # column_descriptions.append({"role": "system", "content": ""})
    print(column_descriptions)

    # model
    messages=(
        [
            SystemMessage(content=prompt_gemini_suggest_plot(column_descriptions)),
            HumanMessage(content="Give me the best recommended plot."),
        ]
    )

    # Invoke Gemini to suggest the plot type
    result = gemini_model.invoke(messages)

    # Extract the recommended plot type from Gemini's response
    recommended_plot_type = result.content.strip()

    return recommended_plot_type





# Function to generate Python code for visualization using Plotly
@st.cache_data
def generate_code_gemini(modified_sql_query, data_description, chart_type):
    messages = [
        SystemMessage(
            content="""
             You are a powerful code generator assistant who can generate codes for visualization based on the modified sql query. 
            
            INSTRUCTIONS:
            1. Your task is to write python code using plotly library for visualization based on the modified SQL query.
            2. Follow the data description for reference.
            3. Apply the given format when generating the code.
            4. Use the given chart types when writing the codes.
            5. Always give different colors to the each values if the number of unique value of each columns is less than 15.
            6. You can sort first the data when plotting line plot.
            7. Always use correct column names from the data_description.
            8. Always generates an error free codes.
            9. Make sure colors are discrete.

            Chart_type: {chart_type}

            FORMAT:
            ```
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
            def visualization(data: pd.DataFrame):
                colors = <AI generated color codes>
                fig = <AI generated codes here>
                return fig
            ```

            DATA DESCRIPTION:
            data_description: {data_description}

            MODIFIED SQL QUERY:
            modified sql query: {modify_query}

            Additional instructions:
            Always output the working codes.

            Output:
            
            """
        ),
        HumanMessage(
            content=f"Generate Python code for the modified SQL query: {modified_sql_query} based on the data description: {data_description} and chart type: {chart_type}"
        ),
    ]

    # Use the Gemini model to generate code
    response = gemini_model.invoke(messages)

    # Extract the generated code
    generated_code = response.content.strip()
    return generated_code

@st.cache_data
def remove_code_wrapper(code_with_language):
    # Assuming the code is wrapped as ```python <code> ```
    start_index = code_with_language.find("```")
    end_index = code_with_language.rfind("```")
    code_without_language = code_with_language[start_index + 3:end_index].strip()

    # Remove the language specification if present
    if code_without_language.lower().startswith("python"):
        code_without_language = code_without_language[len("python"):].strip()

    # Remove triple backticks if present
    if code_without_language.startswith("```") and code_without_language.endswith("```"):
        code_without_language = code_without_language[len("```"): -len("```")].strip()

    return code_without_language


def main():
    st.title("SQL Query Modification and Visualization Code Generation")

    # Input SQL query and question
    input_query = st.text_area("Enter your SQL query:")
    question = st.text_input("Enter a question related to the query:")


    # Modify SQL Query using Gemini
    if st.button("Modify SQL Query"):
        schema = []
        tables = extract_table_names(input_query)
        for table in tables:
            schem = get_table_schema(table)
            schema.append(schem)
        modified_sql_query = modify_sql_query(input_query=input_query, question=question, schema=schema)

        st.text_area("modified_sql_query",modified_sql_query)

        # Get DataFrame based on the modified SQL query
        data_frame = get_data_frame(modified_sql_query)

        # Display DataFrame
        st.write("DataFrame based on Modified SQL Query:")
        st.write(data_frame.head())

        # Generate Visualization Code using Gemini
        data_description = data_description_modified(modified_sql_query)
        print(data_description)
        chart_type = gemini_suggest_plot(data_frame)

        st.text_area("chart type", chart_type)

        code = generate_code_gemini(modified_sql_query, data_description, chart_type)

        # st.plotly_chart(code)
        st.code(code, language="python")

        clean_code = remove_code_wrapper(code)

        code_filename = "generated_code.py"
        with open(code_filename, "w") as code_file:
            code_file.write(clean_code)
        time.sleep(1)

        from generated_code import visualization

        # Assuming 'data_frame' is the DataFrame you want to visualize
        st.plotly_chart(visualization(data_frame))



if __name__ == "__main__":
    main()

