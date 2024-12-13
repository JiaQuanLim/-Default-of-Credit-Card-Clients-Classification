# download_data.py
# author: Shannon Pflueger, Nelli Hovhannisyan, Joseph Lim
# date: 2024-12-4

import click
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--input_data', type=str, help="URL of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")


def main(input_data, write_to):
    """
    Prepare the data and split it into train and test to the specified directory.

    Parameters:
    ----------
    url : str
        The URL of the zip file to be read.
    directory : str
        The directory where the contents of the zip file will be extracted.

    Returns:
    -------
    None
    """

    cc_df = pd.read_csv(input_data)
    cc_df.columns = cc_df.columns.str.replace('_0', '_SEP').str.replace('_2', '_AUG').str.replace('_3', '_JUL').str.replace('_4', '_JUN').str.replace('_5', '_MAY').str.replace('_6', '_APR')
    cc_df.columns = cc_df.columns.str.replace('1', '_SEP').str.replace('2', '_AUG').str.replace('3', '_JUL').str.replace('4', '_JUN').str.replace('5', '_MAY').str.replace('6', '_APR')
    cc_df.rename(columns={'default.payment.next.month': 'target'}, inplace=True)
    cc_df.to_csv(os.path.join(write_to, "processed_data.csv"), index=False)


    train_df, test_df = train_test_split(cc_df, test_size=0.2, random_state=123)
    train_df.to_csv(os.path.join(write_to, "train_df.csv"), index=False)
    test_df.to_csv(os.path.join(write_to, "test_df.csv"), index=False)

if __name__ == '__main__':
    main()