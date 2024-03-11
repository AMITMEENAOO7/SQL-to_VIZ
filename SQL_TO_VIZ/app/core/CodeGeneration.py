# importing libraries
from typing import List
import mysql.connector
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
from .. import config as cf

# initialize the config
config = cf.Config()

# mysql engine
engine = config.engine


# create a cursor
# cursor = engine.cursor()

# defining the function to get the dataframe from sql query
def get_df(query):
    try:
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        return f"Error occured: {str(e)}"

# data description
def data_description_modified(modify_query):
    try:
        df = get_df(modify_query)
        columns = df.columns
        description = []
        for column in columns:
            col = []
            unique = df[column].unique()
            col_description = f"Column_Name: {column}, sample_data: {list(unique[:3])}, number_of_unique_values: {len(unique)}, number_of_rows: {len(df)}, dtype: {unique[0].__class__.__name__}"
            col.append(col_description)
            description.append(col)
        return description
    except Exception as e:
        return f"Error occured: {str(e)}"



# defined the class for structured output
class ChartType(BaseModel):
    DataDescription: List[str] = Field(description="description of the table")
    Question: str = Field(description="question from the user")
    Output: List[str] = Field(description="list of chart types")



# defining a function to determine chart type    
def suggest_plot(data_description, question):
    try:
        # template for system
        parser = JsonOutputParser(pydantic_object=ChartType)

        template= """
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
                Description: {data_description}

                Question: {question}
                """
        system_prompt = PromptTemplate(
            template=template,
            input_variables=["data_description", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )  
        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt) # system prompt
        
        human_prompt = "Give the list of chart types based on the description"

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)  # human prompt
        
        messages = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        fine_tuned_model = ChatOpenAI(
        temperature=0, model_name=config.model, model_kwargs={"top_p": 0}) # fine tuned the model
        # messages = messages.to_messages()
        chain = messages|fine_tuned_model|parser  # adding messages to the model
        response = chain.invoke({"data_description": data_description, "question": question}) # adding messages to the model
        return response
    except Exception as e:
        return f"Error occured: {str(e)}"

# defining functions to generate code based on sql query
def code_generator(modify_query, data_description, chart_type):
    try:
        # template for the system
        template = """
                You are a powerful code generator assistant who can generate codes for visualization based on the modified sql query. 
                
                INSTRUCTIONS:
                1. Your task is to write python code using plotly library for visualization based on the modified SQL query.
                2. Follow the data description for reference.
                3. Apply the given format when generating the code.
                4. Use the given chart types when writing the codes.
                5. Always give different colors to each values and make sure that it gives correct colors.
                6. You can sort first the data when plotting line plot.
                7. Always use correct column names from the data_description.
                8. Always generates an error free codes.
                9. You have to add codes to change the background of the plot.

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
        system_prompt = PromptTemplate(
            template=template,
            input_variables=["chart_type", "data_description", "modify_query"]
        ) 
        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt) # system prompt
        
        human_prompt = "Write a python code for visualization using the given data"

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)  # human prompt
        
        messages = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        messages = messages.format_prompt(modify_query=modify_query, data_description=data_description, chart_type=chart_type)  # add the inputs

        fine_tuned_model = ChatOpenAI(
        temperature=0.2, model_name=config.model) # fine tuned the model

        response = fine_tuned_model.invoke(messages.to_messages())  # adding messages to the model
        return response.content
    except Exception as e:
        return f"Error occured: {str(e)}"
    


if __name__=="__main__":
    modified_query = input("Please input modified query: ")
    question = input("Please input question: ")
    data_description = data_description_modified(modified_query)
    chart_type = suggest_plot(data_description=data_description, question=question)['Output']
    code = code_generator(modify_query=modified_query, data_description=data_description, chart_type=chart_type[0])
    print(code)
