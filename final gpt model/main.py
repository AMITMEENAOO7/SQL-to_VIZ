# Importing libraries
import streamlit as st
import time
from gpt_codegen import get_df, code_generator, data_description_modified, suggest_plot
from modified_sql_2 import modified_sql_query, extract_table_names, get_table_schema

# adding cache data
@ st.cache_data
def get_df_st(query):
    return get_df(query)

@st.cache_data
def code_generator_st(modify_query, data_description, chart_type):
    return code_generator(modify_query, data_description, chart_type)

@st.cache_data
def data_description_modified_st(modify_query):
    return data_description_modified(modify_query)

@st.cache_data
def suggest_plot_st(description, question):
    return suggest_plot(description, question)

@st.cache_data
def modified_sql_query_st(query, question, schema):
    return modified_sql_query(query, question, schema)

@st.cache_data
def extract_table_names_st(query):
    return extract_table_names(query)

@st.cache_data
def get_table_schema_st(table):
    return get_table_schema(table)



# initialized the session state
if "query" and "question" not in st.session_state:
    st.session_state["query"] = None
    st.session_state["question"] = None
    st.session_state["GO"] = False
    st.session_state["modify_query"] = None
    st.session_state["chart_type"] = None
    st.session_state["data"] = None
    st.session_state["option"] = None
    st.session_state["tables"] = None
    st.session_state["schema"] = None
    st.session_state["data_description"] = None
    st.session_state["generated_code"] = None
    st.session_state["plot_type"] = None


# title

st.title("Code Generation And Visualization")

# Input SQL query and questions
query = st.text_area("Input SQL Query", height=50)
question = st.text_input("Question")


st.session_state["query"] = query  # add to session state
st.session_state["question"] = question  # add to session state

execute_query = st.button("GO!!")  # button

if execute_query or st.session_state["GO"]:
    st.session_state["GO"] = True  # set to true if this is ran once
    st.session_state["tables"] = extract_table_names_st(st.session_state["query"])  # getting the table names
    # tables = st.session_state["tables"]
    # print("tables: ", tables)

    # schema
    st.session_state["schema"] = []
    for table in st.session_state["tables"]:
        schem = get_table_schema_st(table)
        st.session_state["schema"].append(schem)
    
    
    schema = st.session_state["schema"]

    mod_query = modified_sql_query_st(query=st.session_state["query"], question=st.session_state["question"], schema=st.session_state['schema'])
    

    
    # display the schema
    st.text_area("Schema", mod_query["Table_schema"]) ## display the schema

    # modified sql query
    st.session_state["modify_query"] = mod_query["Output"]
    modify_query = st.session_state["modify_query"]   # add session state
    

    st.subheader("Modified SQL Query")
    st.code(modify_query, language='sql') # write a text output in streamlit

    # data
    st.session_state["data"] = get_df_st(st.session_state["modify_query"]) # data of from the modified query
    data = st.session_state["data"]
    st.subheader("Data")
    st.dataframe(data, use_container_width=False)

    # data description
    st.session_state["data_description"] = data_description_modified_st(st.session_state["modify_query"]) # data description
    data_description = st.session_state["data_description"] 

    # st.text_area("data description", data_description)
    st.session_state["plot_type"] = suggest_plot_st(description=st.session_state["data_description"], question=st.session_state["question"])
    # testing
    
    # chart types
    st.session_state["chart_type"] = st.session_state["plot_type"]['output']
    

    # sidebar to select the plot types
    with st.sidebar:
        st.session_state["option"] = st.selectbox("Choose plot type", st.session_state["chart_type"], index = 0)
        option = st.session_state["option"]
    
    # time the code generation 
    start_time = time.time()     # add time

    # generate codes  
    st.session_state["generated_code"] = code_generator_st(modify_query=st.session_state["modify_query"], data_description=st.session_state["data_description"], chart_type=option) # generated code
    end_time = time.time()
    generated_code = st.session_state["generated_code"]


    # modifying the code
    end_string = "return fig"

    # remove all unnecessary characters
    start_index = generated_code.find("import")
    generated_code=generated_code[start_index:] 
    end_index = generated_code.find(end_string)
    code = generated_code[:end_index + len(end_string)] # this code might contain some unicode which can't be decoded
    
    # remove characters
    char = ['°']
    for ch in char:
        code = code.replace(ch, '')

    #st.session_state["code"] = code

    # Saving the codes
    with open("gpt_codes_2.py", "w") as f:
        f.write(code)
    
    time.sleep(1)  # this is needed  to allow the file to be saved.
    
    
    # Visualize the data
    from gpt_codes_2 import visualization

    # plot
    st.subheader("Plot")
    st.plotly_chart(visualization(st.session_state['data']))

    # code
    st.subheader("code")
    st.code(code, language='python')

    # Time
    st.subheader("Code generation time")
    st.write((end_time-start_time), " seconds")

