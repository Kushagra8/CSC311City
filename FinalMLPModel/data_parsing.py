import pandas as pd
import numpy as np

# retreive - get
# data - file
# indicator - dummy
# indicators - dummies
# data.drop(col, axis=1, inplace=True) - del data[col]

'''
This code snippet reads a dataset from a CSV file, preprocesses it, 
and prepares the data for training a machine learning model.
'''

data_set = "clean_dataset.csv"

# initialize the internal random number generator
random_state = 42

'''
Preprocessing Functions: Several functions (to_numeric, retreive_number_list, 
retreive_number_list_clean, retreive_number, find_area_at_rank, cat_in_s, 
create_indicators_with_all_categories, normalize_column) are defined to 
preprocess the data. These functions handle tasks such as converting 
strings to numeric values, extracting numbers from strings, creating 
indicator variables for categorical features, and normalizing numeric columns.
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

def create_indicators_with_all_categories(series, prefix, all_categories):
    '''
    create_indicators_with_all_categories creates indicator variables for the given 
    series with the specified prefix and all the possible categories.
    '''
    # retreive indicator variables for the series with the specified prefix
    indicators = pd.get_dummies(series, prefix=prefix)
    
    # Generate column names for all possible categories
    expected_columns = []
    for cat in all_categories:
        expected_columns.append(f"{prefix}_{cat}")
    
    # Reindex the indicator variables with all possible categories
    indicators = indicators.reindex(columns=expected_columns, fill_value=0)
    
    return indicators

def normalize_column(data, column_name):
    '''
    normalize_column normalizes given column.
    '''
    min_val = data[column_name].min()
    max_val = data[column_name].max()
    data[column_name] = (data[column_name] - min_val) / (max_val - min_val)

'''
retreive_data Function: This function reads the CSV file, applies preprocessing 
to the numeric fields, converts categorical features into indicator variables, 
and prepares the features (x) and labels (y) for training the model.
It also splits the data into training and testing sets.
'''

def retreive_data():

    data = pd.read_csv(data_set)

    q1_q4_categories = [-1, 1, 2, 3, 4, 5]
    q6_ranks = q1_q4_categories + [6]

    # data set splits
    n_train = 1200

    # Apply preprocessing to numeric fields to columns Q 7,8,9
    for col in ['Q7', 'Q8', 'Q9']:
        data[col] = data[col].apply(to_numeric).fillna(0)

    # Normalize Q 7,8,9 columns in DataFrame
    for col in ['Q7', 'Q8', 'Q9']:
        normalize_column(data, col)

    # Apply retreive_number function to columns Q 1,2,3,4
    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
        data[col] = data[col].apply(retreive_number)

    # extract list of int from the strings of each element in Q6
    #  result stored back into Q6, replacing original string vals
    #  with int lists.
    data['Q6'] = data['Q6'].apply(retreive_number_list)

    # generates indicator variables for each rank & merges to data
    for i in range(1, 7):
        col_name = f"rank_{i}"
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
        indicators = create_indicators_with_all_categories(data[col_name], col_name, q6_ranks)
        data = pd.concat([data, indicators], axis=1)
    data.drop("Q6", axis=1, inplace=True)

    # for Q 1,2,3,4, category indicators concatenates them with Data, and 
    #  deletes original columns.
    for col in ["Q1", "Q2", "Q3", "Q4"]:
        indicators = create_indicators_with_all_categories(data[col], col, q1_q4_categories)
        data = pd.concat([data, indicators], axis=1)
        data.drop(col, axis=1, inplace=True)

    # Create multi-category indicators
    for cat in ["Partner", "Friend", "Sibling", "Co-worker"]:
        cat_name = f"Q5_{cat}"
        data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    # Remove the original column "Q5"
    data.drop("Q5", axis=1, inplace=True)

    # adds Label to the col specified elbow and then mixes order randomly
    selected_columns = []
    for col in data.columns:
        if col.startswith(('Q1_', 'Q2_', 'Q3_', 'Q4_', 'Q5', 'Q7', 'Q8', 'Q9', 'rank_')):
            selected_columns.append(col)
    selected_columns.append("Label")
    data = data[selected_columns]

    # Shuffle Data's rows randomly
    data = data.sample(frac=1, random_state=42)

    # Extract x features and y labels
    x = data.drop("Label", axis=1).values
    y = pd.get_dummies(data["Label"]).values

    # Splitting data into train & test set
    x_test = x[n_train:]
    y_test = y[n_train:]
    x_train = x[:n_train]
    y_train = y[:n_train]

    return x_train, y_train, x_test, y_test


'''
retreive_file_data Function: This function is similar to retreive_data, but it only 
preprocesses the data without splitting it into training and testing sets. 
It's designed to be used for obtaining data from a file without labels for 
inference or prediction tasks.
'''

def retreive_file_data(file_name):

    data = pd.read_csv(file_name)

    q1_q4_categories = [-1, 1, 2, 3, 4, 5]
    q6_ranks = q1_q4_categories + [6]

    # columns in numeric type, filled missing values with 0's
    for col in ['Q7', 'Q8', 'Q9']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # normaliza
    for col in ['Q7', 'Q8', 'Q9']:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Convert columns Q1-Q4 to their first nums
    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
        data[col] = data[col].apply(retreive_number)

    # Process Q6 to create area rank categories
    data['Q6'] = data['Q6'].apply(retreive_number_list)

    for i in range(1, 7):
        col_name = f"rank_{i}"
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
        indicators = create_indicators_with_all_categories(data[col_name], col_name, q6_ranks)
        data = pd.concat([data, indicators], axis=1)
    data.drop("Q6", axis=1, inplace=True)

    for col in ["Q1", "Q2", "Q3", "Q4"]:
        indicators = create_indicators_with_all_categories(data[col], col, q1_q4_categories)
        data = pd.concat([data, indicators], axis=1)
        data.drop(col, axis=1, inplace=True)

    # Create multi-category indicators
    for cat in ["Partner", "Friend", "Sibling", "Co-worker"]:
        cat_name = f"Q5_{cat}"
        data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    data.drop("Q5", axis=1, inplace=True)

    selected_columns = []
    for col in data.columns:
        if col.startswith(('Q1_', 'Q2_', 'Q3_', 'Q4_', 'Q5', 'Q7', 'Q8', 'Q9', 'rank_')):
            selected_columns.append(col)
    data = data[selected_columns]
    data = data.sample(frac=1, random_state=42)
    return data.values

'''
Main Function Call: The last line of the code calls the retreive_file_data 
function with the path to a CSV file as an argument, which triggers the 
data preprocessing steps and returns the processed data.
'''

retreive_file_data("./clean_dataset.csv")

'''
Overall, this code prepares a dataset for training a machine learning model 
by performing various preprocessing steps such as data cleaning, feature 
extraction, and transformation.
'''