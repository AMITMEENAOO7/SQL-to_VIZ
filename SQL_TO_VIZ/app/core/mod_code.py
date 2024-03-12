import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
from sql_metadata import Parser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from .. import config as cf

# initialize the config
config = cf.Config()

# mysql engine
engine = config.engine

# Gemini model for SQL query modification
gemini_model = ChatGoogleGenerativeAI(model=config.model, convert_system_message_to_human=True)


# Tool to get the table schema

def extract_table_names(query):
    try:
        tables = Parser(query).tables
        return tables
    except:
        return "error in parsing sql query"



def get_table_schema(table):
    '''This function takes in table and output the SQL schema'''
    try:
        query = f'''show create table {table}'''
        table_schema = pd.read_sql_query(query, engine)
        schema = table_schema['Create Table'].iloc[0]
        index = schema.find("ENGINE")
        schema = schema[:index]
        return schema
    except Exception as e:
        return f"Error occured: {str(e)}"


# defined the class for structured output
class Mod_SQL(BaseModel):
    Input_query: str = Field(description="input sql query")
    Question: str = Field(description="user's question")
    Table_schema: str = Field(description="schema of the tables")
    Thoughts: str = Field(description="thoughts of the model")
    Action: str = Field(description="Action of the model")
    Observation: str = Field(description="Observation by the model")
    Output: str = Field(description="output sql query")



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
        gemini = prompt | gemini_model | parser

        response = gemini.invoke({"input_query": input_query, "question": question, "schema": schema,
                                  "format_instructions": format_instructions})
        return response["Output"]

    except Exception as e:
        return f"Error modifying SQL query: {e}"



def get_data_frame(modified_sql_query):
    try:
        # Execute the modified SQL query and get DataFrame
        df = pd.read_sql_query(modified_sql_query, engine)
        return df
    except Exception as e:
        return f"Error executing modified SQL query: {e}"



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



def gemini_suggest_plot(question, data_descriptions):

    try:
        # parser
        parser = JsonOutputParser(pydantic_object=ChartType)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate.from_template(template_gemini_suggest_plot)
        prompt = prompt.partial(format_instructions=format_instructions)

        # chain
        chain = prompt | gemini_model | parser
        response = chain.invoke(
            {"data_descriptions": data_descriptions, "question": question})  # adding messages to the model
        return response['output']

    except  Exception as e:
        return f"Error occured: {str(e)}"


# data description

def data_description_modified(modify_query):
    try:
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
    except  Exception as e:
        return f"Error occured: {str(e)}"


# Function to generate Python code for visualization using Plotly

def generate_code_gemini(modified_sql_query, data_descriptions, chart_type):
    try:
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


        return response.content


    except  Exception as e:
        return f"Error occured: {str(e)}"



# if __name__=="__main__":
#     modified_query = '''SELECT c.name AS category, COUNT(fc.film_id) AS num_films
#                     FROM film_category fc
#                     JOIN category c ON fc.category_id = c.category_id
#                     GROUP BY c.name'''
#     data = get_data_frame(modified_query)
#     print(data)
# #     # data_descripition = data_description_modified(modified_query)
# #     # chart = gemini_suggest_plot(question, data_descripition)
# #     # code = generate_code_gemini(modified_query,data_descripition,chart)
# #     #
# #     # print(chart)
# #     # print(code)
