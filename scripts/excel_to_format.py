"""Converts the generated/labeled data in the archive to training formats.

Converted and ready-for-training files go to the main directory.
However, the original files are kept in the archive for the sake of organization.

Consider a naming format to keep track of the archive files.
Maybe delete older training data in the main directory to minimize mess.

Author: Mohamed Mostafa
"""

import pandas as pd
from sklearn.model_selection import train_test_split

path = '/workspace/' # Main folder
fake_file_name = 'archive/FILE_NAME.xlsx'
real_file_name = 'archive/FILE_NAME.csv'

def fake_excel_tsv(directory, name):
    """
    Read fake excel file and convert it to TSV format.

    Args:
        directory (str): Directory path of the file.
        name (str): Name of the file.
    """
    
    df_fake = pd.read_excel(directory + name)
    # df_fake = pd.read_csv(directory + name)
    # df_fake = df_fake[df_fake['essay_set'] == 1] #Choose essay/prompt set/id if several
    
    # Split the data into train, validation, and test sets
    fake_train, fake_test_val = train_test_split(df_fake, test_size=0.3, random_state=42)
    fake_valid, fake_test = train_test_split(fake_test_val, test_size=0.3, random_state=42)
    
    # Convert the dataframes to TSV format
    to_tsv(df_fake, fake_train, directory, 'data/fake_train')
    to_tsv(df_fake, fake_valid, directory, 'data/fake_valid')
    to_tsv(df_fake, fake_test, directory, 'data/fake_test')
    
def real_excel_tsv(directory, name):
    """
    Read real excel file and convert it to TSV format.

    Args:
        directory (str): Directory path of the file.
        name (str): Name of the file.
    """   
    
    df_real = pd.read_csv(directory + name)
    # df_real = pd.read_excel(directory + name)
    # df_real = df_real[df_real['essay_set'] == 1] #Choose essay/prompt set/id if several
    
    # Split the data into train, validation, and test sets
    real_train, real_test_val = train_test_split(df_real, test_size=0.3, random_state=42)
    real_valid, real_test = train_test_split(real_test_val, test_size=0.3, random_state=42)
    
    # Convert the dataframes to TSV format
    to_tsv(df_real, real_train, directory, 'data/real_train')
    to_tsv(df_real, real_valid, directory, 'data/real_valid')
    to_tsv(df_real, real_test, directory, 'data/real_test')
    
def to_tsv(df, set, directory, name):
    """
    Convert dataframe to TSV format and write it to a file.

    Args:
        df (pd.DataFrame): Dataframe to be converted.
        set (pd.DataFrame): Subset of the dataframe.
        directory (str): Directory path to save the file.
        name (str): Name of the file.
    """
    
    #Replace all columns having spaces with underscores
    set.columns = [c.replace(' ', '_') for c in df.columns]

    #Replace all fields having line breaks with space
    set = set.replace('\n', ' ',regex=True)

    #Write dataframe into TSV
    set.to_csv(directory + name + '.tsv', sep='\t', encoding='utf-8',  index=False, lineterminator='\r\n')
    
def fake_excel_jsonl(directory, name):
    """
    Read fake excel file and convert it to JSONL format.

    Args:
        directory (str): Directory path of the file.
        name (str): Name of the file.
    """
    
    df_fake = pd.read_excel(directory + name)
    # df_fake = pd.read_csv(directory + name)
    # df_fake = df_fake[df_fake['essay_set'] == 1] #Choose essay/prompt set/id if several
    
    # Split the data into train, validation, and test sets
    fake_train, fake_test_val = train_test_split(df_fake, test_size=0.3, random_state=42)
    fake_valid, fake_test = train_test_split(fake_test_val, test_size=0.3, random_state=42)
    
    # Save the train, validation, and test sets to JSONL files
    fake_train.to_json(directory + 'data/fake_train.jsonl', orient="records", lines="True")
    fake_test.to_json(directory + 'data/fake_test.jsonl', orient="records", lines="True")
    fake_valid.to_json(directory + 'data/fake_valid.jsonl', orient="records", lines="True")
    
def real_excel_jsonl(directory, name):
    """
    Read real excel file and convert it to JSONL format.

    Args:
        directory (str): Directory path of the file.
        name (str): Name of the file.
    """
    
    df_real = pd.read_csv(directory + name)
    # df_real = pd.read_excel(directory + name)
    # df_real = df_real[df_real['essay_set'] == 1] #Choose essay/prompt set/id if several
    
    # Split the data into train, validation, and test sets
    real_train, real_test_val = train_test_split(df_real, test_size=0.3, random_state=42)
    real_valid, real_test = train_test_split(real_test_val, test_size=0.3, random_state=42)
    
    # Save the train, validation, and test sets to JSONL files
    real_train.to_json(directory + 'data/real_train.jsonl', orient="records", lines="True")
    real_valid.to_json(directory + 'data/real_valid.jsonl', orient="records", lines="True")
    real_test.to_json(directory + 'data/real_test.jsonl', orient="records", lines="True")

# example execution, uncomment, for testing

# real_excel_tsv(path, real_file_name)
# fake_excel_tsv(path, fake_file_name)

# real_excel_jsonl(path, real_file_name)
# fake_excel_jsonl(path, fake_file_name)