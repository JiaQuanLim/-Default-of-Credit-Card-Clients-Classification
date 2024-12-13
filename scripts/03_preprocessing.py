# eda.py
# author: Shannon Pflueger, Nelli Hovhannisyan, Joseph Lim
# date: 2024-12-4

import click
import os
import numpy as np
import pandas as pd
import altair_ally as aly
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer

@click.command()
@click.option('--input_data', type = str, help = "Path to processed training data")
#@click.option('--plot_to', type = str, help = "Path to directory where the plot will be written to")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
@click.option('--preprocessor_to', type = str, help = "Path to directory where the preprocessor will be written to")


def main(input_data, write_to, preprocessor_to):
    '''
    Main function to perform EDA (Exploratory Data Analysis) on the provided dataset.

    Parameters:
    -----------
    processed_training_data : str
        Path to the processed training data CSV file.
    
    plot_to : str
        Directory path where the generated plots (heatmap, countplot, boxplots) 
        will be saved.
    
    table_to : str
        Directory path where the generated summary tables (info, describe, shape)
        will be saved.

    Returns:
    --------
    None
        The function saves plots and tables as files in the specified directories.
    '''

    train_df = pd.read_csv(input_data)
    
    # Combining education group 6 to group 5 as both belongs to the same grouping
    train_df.loc[train_df.EDUCATION == 6, 'EDUCATION'] = 5

    X_train, y_train = train_df.drop(columns=['target']), train_df['target']
    X_train.to_csv(os.path.join(write_to, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(write_to, "y_train.csv"), index=False)

    binary_features = ['SEX']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    drop_features = ['ID']
    numeric_features = ['LIMIT_BAL', 'AGE',
                        'PAY_SEP', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR', 
                        'BILL_AMT_SEP', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY', 'BILL_AMT_APR',
                        'PAY_AMT_SEP', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR']
    
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("drop", drop_features),
        verbose_feature_names_out= False
    )
    
    with open(os.path.join(preprocessor_to, "preprocessor.pickle"), 'wb') as f:
        pickle.dump(preprocessor, f)
    

if __name__ == '__main__':
    main()