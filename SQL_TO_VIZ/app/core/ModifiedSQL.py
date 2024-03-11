# importing libraries
from typing import List
import mysql.connector
from sql_metadata import Parser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
from .. import config as cf


# initialized the config
config = cf.Config()

# mysql engine
engine = config.engine

# create a cursor
# cursor = engine.cursor()


# function to extract the tables names
def extract_table_names(query):
    try:
        tables = Parser(query).tables
        return tables
    except:
        return "error in parsing sql query"


# tool to get the table description
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
        print(f"An error occurred: {str(e)}")
        return None



# defined the class for structured output
class Mod_SQL(BaseModel):
    Input_query: str = Field(description="input sql query")
    Question: str = Field(description="user's question")
    Table_schema: List[str] = Field(description="schema of the tables")
    Thoughts: str = Field(description="thoughts of the model")
    Action: str = Field(description="Action of the model")
    Observation: str = Field(description="Observation by the model")
    Output: str = Field(description="output sql query")


# function to get the modified sql query
def modified_sql_query(query, question, schema):
    try:
        # parser
        parser = JsonOutputParser(pydantic_object=Mod_SQL)
        # template for the system
        template = """
                System:
                You are a powerful assistant specialized in modifying the SQL queries for effective data visualization.

                Instructions:
                1. Modify the SQL query in such a way that it can provides data for visualization.
                2. Use the input query and the question when modifying the query.
                3. Use the Table schema to get the column and its type when modifying the query.
                4. Modify the query but keep the original query intact as much as possible but should always answer the user question.
                5. The model should always output the modified SQL query enclosed within triple backticks in the specified format.
                
                ALways use the following thought process while generating SQL queries:
                Thoughts: here the model should think about how to modify the SQL query.
                Action: here the model should generate the modified query.
                Observations: here the model should observe if the modified query it generated is valid and if it answer the user question.If not you can run again till you reached the answer.
                Final: I now know the answer.
                Output: ``` ```

                Always use the following format when output the values:
                {format_instructions}

                
                Example:
                Input_query: '''SELECT crop, crop_type from crop_production where state_name = 'Telangana';'''
                Question: "what is the average yield in the state Telangana"
                Model_Output: '''SELECT crop, avg(yield) as average_yield from crop_production where state_name= 'Telangana' GROUP BY crop ORDER BY crop;''

                Inputs:
                SQL_query: {query}

                Question: {question}

                Table schema: {schema}
                """
        system_prompt = PromptTemplate(
            template=template,
            input_variables=["query", "question", "schema"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            ) 
        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt) # system prompt
        
        human_prompt = "Give me the modified SQL query"

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)  # human prompt
        
        messages = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        # print(messages)
        fine_tuned_model = ChatOpenAI(
        temperature=0, model_name=config.model, model_kwargs={"top_p":0}) # fine tuned the model

        chain = messages|fine_tuned_model|parser  # adding messages to the model
        response = chain.invoke({"query": query, "question": question, "schema": schema})
        return response
    except Exception as e:
        print(f"An error occured: {str(e)}")


# test the codes
if __name__=="__main__":
    query = input("please input query")
    question = input("question")
    tables = extract_table_names(query)
    schema = [get_table_schema(table) for table in tables]
    modified_SQL = modified_sql_query(query, question, schema)
    print(modified_SQL["Output"])
