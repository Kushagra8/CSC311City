import re
import pandas as pd

'''
This code snippet reads a dataset from a CSV file, preprocesses it, 
and prepares the data for training a machine learning model.
'''

file_name = "clean_dataset.csv"

# initialize the internal random number generator
random_state = 42

'''
Preprocessing Functions: Several functions (to_numeric, retreive_number_list, 
retreive_number_list_clean, retreive_number, find_area_at_rank, cat_in_s, 
create_dummies_with_all_categories, normalize_column) are defined to 
preprocess the data. These functions handle tasks such as converting 
strings to numeric values, extracting numbers from strings, creating 
dummy variables for categorical features, and normalizing numeric columns.
'''

def find_area_at_rank(l, i):
    '''
    find_area_at_rank returns area at a certain rank in list `l`.
    return -1 iif area is not in `l`.
    '''
    if i in l:
        return l.index(i) + 1
    else:
        return -1
    
def cat_in_s(s, cat):
    '''
    cat_in_s returns cat present in string `s`
    returns 0 if cat is not found in s
    '''
    if not pd.isna(s):
        return int(cat in s)
    else:
        return 0

def to_numeric(s):
    '''
    to_numeric converts string s to float.
    Invalid & NaN converted to float('nan').
    '''

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def retreive_number_list(s):
    """
    retreive_number_list retrieves list of integers from string `s`.
    Pad list with -1 until the list's len is 6.
    """

    numbers = []
    for char in str(s):
        if char.isdigit():
            numbers.append(int(char))
    
    # Pad the list with -1 until its length is 6
    while len(numbers) < 6:
        numbers.append(-1)
    
    return numbers

def retreive_number(s):
    '''
    retreive_number retreives first numb in string `s`.
    And return -1 if it does not contain any numbers.
    '''
    
    n_list = retreive_number_list(s)
    if len(n_list) >= 1:
        return n_list[0]
    else:
        return -1

def create_dummies_with_all_categories(series, prefix, all_categories):
    '''
    create_dummies_with_all_categories creates dummy variables for the given 
    series with the specified prefix and all the possible categories.
    '''
    # Get dummy variables for the series with the specified prefix
    dummies = pd.get_dummies(series, prefix=prefix)
    
    # Generate column names for all possible categories
    expected_columns = []
    for cat in all_categories:
        expected_columns.append(f"{prefix}_{cat}")
    
    # Reindex the dummy variables with all possible categories
    dummies = dummies.reindex(columns=expected_columns, fill_value=0)
    
    return dummies

def normalize_column(data, column_name):
    '''
    normalize_column normalizes given column.
    '''
    min_val = data[column_name].min()
    max_val = data[column_name].max()
    data[column_name] = (data[column_name] - min_val) / (max_val - min_val)


'''
retreive_data Function: This function reads the CSV file, applies preprocessing 
to the numeric fields, converts categorical features into dummy variables, 
and prepares the features (x) and labels (y) for training the model.
It also splits the data into training and testing sets.
'''


'''
retreive_file_data Function: This function is similar to retreive_data, but it only 
preprocesses the data without splitting it into training and testing sets. 
It's designed to be used for obtaining data from a file without labels for 
inference or prediction tasks.
'''


'''
Main Function Call: The last line of the code calls the retreive_file_data 
function with the path to a CSV file as an argument, which triggers the 
data preprocessing steps and returns the processed data.
'''


'''
Overall, this code prepares a dataset for training a machine learning model 
by performing various preprocessing steps such as data cleaning, feature 
extraction, and transformation.
'''