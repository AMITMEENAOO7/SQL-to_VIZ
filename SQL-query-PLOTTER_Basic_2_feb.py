import json

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import openai
import ast
import plotly.express as px
import re

# Set up OpenAI API key
openai.api_key = "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"


# Function to execute SQL query and retrieve data
def execute_sql_query(query, connection):
    try:
        result = pd.read_sql_query(query, connection)
        return result
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")


# Function to generate prompt for OpenAI
# Function to generate prompt for OpenAI
def generate_prompt(result):
    prompt = """Based on the provided data, please suggest the best plot type and parameters for visualization. Additionally, provide the names for the x-axis and y-axis in the following json format: 

    {"plot": "plot_type", "x_axis": "name_of_x_axis", "y_axis": "name_of_y_axis"}.

    For example:
    if the sql query is : SELECT State_Name, Production_in_tons FROM crop_production;
    then the result should be :
    {"plot": "bar", "x_axis": "State_Name", "y_axis": "Production_in_tons"}.

    Please ensure that the x-axis and y-axis names are clearly identified. Do not generate column names of your own. Give from the provided data only.

    The table schema is as follows:
    """
    for i, col in enumerate(result.columns):
        prompt += f"{i + 1}. {col}\n"
    return prompt


def main():
    st.title('SQL Query Plotter')

    # Database connection setup
    db_user = 'root'
    db_password = '00000000'
    db_host = 'localhost'
    db_port = '3306'
    db_name = 'viz'

    # Construct database URI using mysql-connector-python
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    engine = create_engine(db_uri)
    connection = engine.connect()

    # SQL query input
    query = st.text_area('Enter your SQL query:', height=200)

    # Execute SQL query
    if st.button('Run Query'):
        result = execute_sql_query(query, connection)
        st.write('Query Result:')
        st.write(result)

        # Generate prompt for selecting suitable parameters
        prompt = generate_prompt(result)

        # Use OpenAI to generate guidance for selecting suitable plot type and parameters
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.1,
            max_tokens=100
        )

        # Process OpenAI response
        suggestion = response.choices[0].text.strip()
        suggestion = re.search(r'\{.*?\}', suggestion)
        if suggestion:
            suggestion = suggestion.group()
            print("\n\n$$\n\n", suggestion)


        suggestion_dict = eval(suggestion)
        print(suggestion_dict)

        # Extract plot type from suggestion
        plot_type = suggestion_dict['plot']

        # Extract x-axis and y-axis from suggestion
        x_axis = suggestion_dict['x_axis']
        y_axis = suggestion_dict['y_axis']

        # Ensure both x and y are extracted
        if x_axis is None or y_axis is None:
            st.error("Unable to extract both x-axis and y-axis from the suggestion.")
        else:
            st.subheader(f'Suggested Plot Type: {plot_type}')
            st.subheader(f'Suggested X-axis: {x_axis}')
            st.subheader(f'Suggested Y-axis: {y_axis}')

            # Print extracted x-axis and y-axis values for debugging
            print("Extracted x-axis:", x_axis)
            print("Extracted y-axis:", y_axis)

            # st.plotly_chart(create_plot(result, plot_type, x_axis, y_axis))
            st.plotly_chart(getattr(px, plot_type)(result, x_axis, y_axis))


if __name__ == "__main__":
    main()
