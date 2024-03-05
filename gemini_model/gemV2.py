import time
import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from sql_metadata import Parser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import mysql.connector
import os
from dotenv import load_dotenv
from typing import List
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


# defined the class for structured output
class Mod_SQL(BaseModel):
    Input_query: str = Field(description="input sql query")
    Question: str = Field(description="user's question")
    Table_schema: str = Field(description="schema of the tables")
    Thoughts: str = Field(description="thoughts of the model")
    Action: str = Field(description="Action of the model")
    Observation: str = Field(description="Observation by the model")
    Output: str = Field(description="output sql query")

@st.cache_data
def modify_sql_query(input_query, question, schema):
    """
    Modifies an SQL query based on a user's question using Gemini.

    Args:
        input_query (str): The original SQL query.
        question (str): The user's question related to the query.

    Returns:
        str: The modified SQL query, or None if an error occurs.
    """
    try:
        # parser
        parser = JsonOutputParser(pydantic_object=Mod_SQL)
        format_instructions = parser.get_format_instructions()
        # Construct the prompt for Gemini
        template = """  You are a powerful assistant specialized in modifying the SQL queries for effective data visualization.

            Instructions:
            1. Modify the SQL query in such a way that it can provides data for visualization.
            2. Use the input query and the question when modifying the query.
            3. Use the Table schema to get the column and its type when modifying the query.
            4. Modify the query but keep the original query intact as much as possible but should always answer the user question.
            5. Always order the SQL query.
            6. Always output only the SQL query without  additional strings.
            7. Always give complete SQL queries.
            
            Always use the following thought process while generating SQL queries:
            Thoughts: here the model should think about how to modify the SQL query.
            Action: here the model should generate the modified query.
            Observations: here the model should observe if the modified query it generated is valid and if it answer the user question.If not you can run again till you reached the answer.
            Final: I now know the answer.
            Output:

            Always use the following format when output the values:
            {format_instructions}

            
            Example:
            Input_query: '''SELECT crop, crop_type from crop_production where state_name = 'Telangana';'''
            Question: "what is the average yield in the state Telangana"
            Model_Output: '''SELECT crop, avg(yield) as average_yield from crop_production where state_name= 'Telangana' GROUP BY crop ORDER BY crop;''

            Inputs:
            SQL_query: {input_query}

            Question: {question}

            Table schema: {schema}
            """
        # prompt
        prompt = PromptTemplate.from_template(template)
        prompt = prompt.partial(format_instructions=format_instructions)

        # Use Gemini to modify the SQL query
        gemini = prompt|gemini_model|parser
        # # Extract the modified SQL query from Gemini's response
        # modified_sql_query = gemini_response.content.strip()
        #
        # # Replace BIGNUMERIC and INT64 with appropriate MySQL types
        # modified_sql_query = modified_sql_query.replace("BIGNUMERIC", "DECIMAL(18,2)")
        # modified_sql_query = modified_sql_query.replace("INT64", "INT")
        response = gemini.invoke({"input_query": input_query, "question": question, "schema": schema, "format_instructions": format_instructions})
        return response["Output"]

    except Exception as e:
        print(f"Error modifying SQL query: {e}")
        return None


@st.cache_data
def get_data_frame(modified_sql_query):
    try:
        # Execute the modified SQL query and get DataFrame
        df = pd.read_sql_query(modified_sql_query, engine)
        return df
    except Exception as e:
        print(f"Error executing modified SQL query: {e}")
        return None


# Define gemini_suggest_plot function


# Prompt template for gemini_suggest_plot
template_gemini_suggest_plot = """
You are a powerful assistant who can help to determine the chart types for visualization from the data description. 
            
            INSTRUCTIONS:
            1. Your task is to determine the chart type for visualization.
            2. Use the column name, sample data, etc., from the data description when determining the chart type.
            3. You can also refer to the user question when determining the chart type.
            4. Always suggest different plots like pie plot or violin plot based on the data.
            5. Output a list of plots suitable for the data.
            6. Give the most suitable plot first.

            This is the example list: ["bar plot", ...)]

            ALways follow the following format when output:
            format: {format_instructions}

            Inputs:
            data_frame_preview: {data_descriptions}

            Question: {question}
"""

# defined the class for structured output
class ChartType(BaseModel):
    description: List[str] = Field(description="description of the table")
    question: str = Field(description="question from the user")
    output: List[str] = Field(description="list of chart types")




@st.cache_data
def gemini_suggest_plot(question,data_descriptions):
    # parser
    parser = JsonOutputParser(pydantic_object=ChartType)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate.from_template(template_gemini_suggest_plot)
    prompt = prompt.partial(format_instructions= format_instructions)

    # chain
    chain= prompt|gemini_model|parser
    response = chain.invoke({"data_descriptions": data_descriptions, "question": question})  # adding messages to the model
    return response['output']


# data description
@st.cache_data
def data_description_modified(modify_query):
    df = get_data_frame(modify_query)
    columns = df.columns
    description = []
    for column in columns:
        col = []
        unique = df[column].unique()
        col_description = f"Column_Name: {column}, sample_data: {list(unique[:3])}, number_of_unique_values: {len(unique)}, dtype: {unique[0].__class__.__name__}"
        col.append(col_description)
        description.append(col)
    return description

# Function to generate Python code for visualization using Plotly
@st.cache_data
def generate_code_gemini(modified_sql_query, data_descriptions, chart_type):
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
            10. Give appropriate title to the visualisation.

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
            data_description: {data_descriptions}

            MODIFIED SQL QUERY:
            modified sql query: {modify_query}

            Additional instructions:
            Always output the working codes.

            Output:

            """
        ),
        HumanMessage(
            content=f"Generate Python code for the modified SQL query: {modified_sql_query} based on the data description: {data_descriptions} and chart type: {chart_type}"
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
    @st.cache_data
    def get_data_frame_st(query):
        return get_data_frame(query)

    @st.cache_data
    def generate_code_gemini_st(modify_query, data_descriptions, chart_type):
        return generate_code_gemini(modify_query, data_descriptions, chart_type)

    @st.cache_data
    def data_description_modified_st(modify_query):
        return data_description_modified(modify_query)

    @st.cache_data
    def gemini_suggest_plot_st(data_descriptions, question):
        return gemini_suggest_plot(data_descriptions, question)

    @st.cache_data
    def modify_sql_query_st(query, question, schema):
        return modify_sql_query(query, question, schema)

    @st.cache_data
    def extract_table_names_st(query):
        return extract_table_names(query)

    @st.cache_data
    def get_table_schema_st(table):
        return get_table_schema(table)

    @st.cache_data
    def remove_code_wrapper_st(code_with_language):
        return remove_code_wrapper(code_with_language)


    if "input_query" not in st.session_state:
        st.session_state["input_query"] = None
        st.session_state["question"] = None
        st.session_state["modified_sql_query"] = None
        st.session_state["selected_plot_type"] = None
        st.session_state["schema"] = None
        st.session_state["data_descriptions"] = None
        st.session_state["chart_type"] = None
        st.session_state["data_frame"] = None
        st.session_state["code"] = None
        st.session_state["SQL_VIZ"] = False







    st.title("SQL Query Modification and Visualization Code Generation")

    with st.form("my_form"):
        # Input SQL query and question
        input_query = st.text_area("Enter your SQL query:")
        question = st.text_input("Enter a question related to the query:")
        st.session_state["query"] = input_query  # add to session state
        st.session_state["question"] = question  # add to session state
        button = st.form_submit_button("Modify SQL Query")




    # Modify SQL Query using Gemini
    if button | st.session_state["SQL_VIZ"]:
        st.session_state["SQL_VIZ"] = True

        schema = []
        tables = extract_table_names(input_query)
        for table in tables:
            schem = get_table_schema(table)
            schema.append(schem)
        start_modify_time = time.time()
        st.session_state.modified_sql_query = modify_sql_query_st(query=input_query, question=question, schema=schema)
        end_modify_time = time.time()

        st.text_area("modified_sql_query", st.session_state.modified_sql_query)

        # Get DataFrame based on the modified SQL query
        start_data_frame_time = time.time()
        st.session_state["data_frame"] = get_data_frame_st(st.session_state.modified_sql_query)
        end_data_frame_time = time.time()

        # Display DataFrame
        data_frame = st.session_state["data_frame"]

        st.write("DataFrame based on Modified SQL Query:")
        st.write(data_frame.head())

        # Generate Visualization Code using Gemini
        st.session_state["data_descriptions"] = data_description_modified_st(st.session_state.modified_sql_query)
        data_descriptions = st.session_state["data_descriptions"]

        start_chart_type_time = time.time()
        st.session_state["chart_type"] = gemini_suggest_plot_st(question = st.session_state["question"], data_descriptions=st.session_state["data_descriptions"])
        chart_type = st.session_state["chart_type"]
        end_chart_type_time = time.time()

        # st.text_area("chart type", chart_type)

        st.session_state["selected_plot_type"] = st.selectbox("Choose chart type",  st.session_state["chart_type"],index = 0)
        selected_plot_type = st.session_state["selected_plot_type"]




        start_code_generation_time = time.time()
        st.session_state["code"] = generate_code_gemini_st(st.session_state.modified_sql_query, data_descriptions, selected_plot_type)
        code = st.session_state["code"]
        end_code_generation_time = time.time()



        st.code(code, language="python")

        clean_code = remove_code_wrapper(code)

        code_filename = "generated_code.py"
        with open(code_filename, "w") as code_file:
            code_file.write(clean_code)
        time.sleep(1)

        start_visualization_time = time.time()

        from generated_code import visualization

        st.plotly_chart(visualization(st.session_state["data_frame"]))
        end_visualization_time = time.time()

        st.write(f"Time taken to modify SQL query: {end_modify_time - start_modify_time:.2f} seconds")
        st.write(f"Time taken to get DataFrame: {end_data_frame_time - start_data_frame_time:.2f} seconds")
        st.write(f"Time taken to suggest chart type: {end_chart_type_time - start_chart_type_time:.2f} seconds")
        st.write(f"Time taken to generate code: {end_code_generation_time - start_code_generation_time:.2f} seconds")
        st.write(f"Time taken to generate visualization: {end_visualization_time - start_visualization_time:.2f} seconds")


if __name__ == "__main__":
    main()

    # Testing
# sql = """SELECT a.first_name, a.last_name
# FROM actor a
# LEFT JOIN film_actor fa ON a.actor_id = fa.actor_id
# LEFT JOIN film f ON fa.film_id = f.film_id AND f.title = 'ACADEMY DINOSAUR'
# GROUP BY a.first_name, a.last_name;"""
# question = "what are the rental rates for all movie titles by Penelope?"
#
# tables = extract_table_names(sql)
#
# schema = []
# for table in tables:
#     schema.append(get_table_schema(table))
# response = modify_sql_query(sql, question, schema)
#
# print(response)
# print(response['Output'])