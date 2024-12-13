# eda.py
# author: Shannon Pflueger, Nelli Hovhannisyan, Joseph Lim
# date: 2024-12-4

import click
import os
import numpy as np
import pandas as pd
import altair_ally as aly

@click.command()
@click.option('--input_data', type = str, help = "Path to processed training data")
@click.option('--plot_to', type = str, help = "Path to directory where the plot will be written to")
#@click.option('--table_to', type = str, help = "Path to directory where the table will be written to")


def main(input_data, plot_to):
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
    
    aly.alt.data_transformers.enable('vegafusion')
    aly.dist(train_df, color='target').save(os.path.join(plot_to, "data_distribution.png"))
    aly.corr(train_df).save(os.path.join(plot_to, "correlation_plot.png"))
    columns_with_at_least_one_high_corr = [
        "PAY_APR",
        "PAY_MAY",
        "PAY_JUN",
        "PAY_JUL",
        "PAY_AUG",
        "PAY_SEP",
        "target", 
    ]
    aly.pair(train_df[columns_with_at_least_one_high_corr].sample(300), color='target').save(os.path.join(plot_to, "pay_correlation.png"))
    columns_with_at_least_one_high_corr = [
        "BILL_AMT_APR",
        "BILL_AMT_MAY",
        "BILL_AMT_JUN",
        "BILL_AMT_JUL",
        "BILL_AMT_AUG",
        "BILL_AMT_SEP",
        "target", 
    ]
    
    aly.pair(train_df[columns_with_at_least_one_high_corr].sample(300), color='target').save(os.path.join(plot_to, "bill_amt_correlation.png"))

if __name__ == '__main__':
    main()