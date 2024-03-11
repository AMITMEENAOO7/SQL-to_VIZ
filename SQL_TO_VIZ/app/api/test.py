from ..utils.utilities import clean_code
# from ..core.ModifiedSQL import extract_table_names

from sql_metadata import Parser
import asyncio
# function to extract the tables names
async def extract_table_names(query):
    try:
        tables = await Parser(query).tables  # Note the use of await here
        return tables
    except Exception as e:
        return f"Error in parsing SQL query: {str(e)}"

code = """
```
import pandas as pd
    def visualization(data: pd.DataFrame):
    colors = px.colors.qualitative.Plotly
    fig = px.bar(data, x='actor_id', y='actor_count', color='actor_id', color_discrete_map={str(i): colors[i] for i in range(len(data['actor_id'].unique()))})
    return fig
```
"""



# function to extract the tables names
def extract_table_names(query):
    try:
        tables = Parser(query).tables
        return tables
    except Exception as e:
        return f"Error in parsing SQL query: {str(e)}"

async def async_extract_table_names(query):
    return extract_table_names(query)

async def main():
    query = "SELECT * FROM my_table"
    tables = await async_extract_table_names(query)
    print(tables)

# Call the main function
asyncio.run(main())
