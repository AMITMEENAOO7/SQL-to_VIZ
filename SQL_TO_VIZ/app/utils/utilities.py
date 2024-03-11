# ALL THE UTILITIES ARE DEFINED HERE

def clean_code(code):
    '''Function to get rid of unwanted characters'''
    start_index = code.find("import")
    end_index = code.find("return fig") + len("return fig")

    # clean code
    clean = code[start_index:end_index]
    return clean