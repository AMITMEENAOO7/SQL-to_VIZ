import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ..core.ModifiedSQL import modified_sql_query, extract_table_names, get_table_schema
from ..core.CodeGeneration import get_df, code_generator, data_description_modified, suggest_plot
from ..config import Config
from ..utils.utilities import clean_code
from pathlib import Path
import asyncio

# Initialize the configuration
config = Config()

# defining the path to save the code file
p = Path("app\\api\\")
# Create FastAPI app instance
app = FastAPI()

# Define Pydantic class for API response
class Outputs(BaseModel):
    table: List[str]
    table_schema: List[str]
    modified_sql: str
    chart_type: List[str]
    code: str
    plot_div: str

# Define root route
@app.get("/")
async def read_root():
    return "HOME PAGE!!!"

# Define query route
@app.post("/query", response_model=Outputs)
async def sql2viz(query:str, question:str):
    try:
        # Extract table names from the query
        tables = extract_table_names(query)
        
        # Get schema for each table
        table_schema = []
        for i, table in enumerate(tables):
            table_schema.append(get_table_schema(table))
        
        # Modify the SQL query
        modified_sql_result = modified_sql_query(query=query, question=question, schema=table_schema)
        modified_sql = modified_sql_result['Output']
        
        # Get data description
        data_description = data_description_modified(modify_query=modified_sql)
        
        # Suggest plot types
        chart_type = suggest_plot(data_description=data_description, question=question)['Output']
    
        # Generate code
        generated_code = code_generator(modify_query=modified_sql, data_description=data_description, chart_type=chart_type[0])  # should put an action where if the chart type is select by the user
        
        # Clean generated code
        code = str(clean_code(generated_code))

        # execute code
        # exec(code)

        with open(p/"code.py", "w") as f:
            f.write(code)
        
        # sleep
        await asyncio.sleep(1)

        # data
        data = get_df(modified_sql)

        # import visualization
        from .code import visualization
        fig = visualization(data)

        # plot div
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Prepare response
        response = Outputs(table=tables, 
                           table_schema=table_schema, 
                           modified_sql=modified_sql, 
                           chart_type=chart_type, 
                           code=code,
                           plot_div=plot_div)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# start API server
def start_api_server(host, port):
    return uvicorn.run("app.api.app:app", host=host, port=port, reload=True, access_log=False)


# Add main block to run the app directly
if __name__ == "__main__":
    start_api_server(host=config.api_host, port=config.api_port)
