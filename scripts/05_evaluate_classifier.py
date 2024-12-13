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
import matplotlib.pyplot as plt

@click.command()
@click.option('--input_data', type = str, help = "Path to processed training data")
#@click.option('--plot_to', type = str, help = "Path to directory where the plot will be written to")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
@click.option('--best_model_from', type=str, help="Path to directory where the preprocessor object lives")


def main(input_data, write_to, best_model_from):
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

    test_df = pd.read_csv(input_data)
    
    # Combining education group 6 to group 5 as both belongs to the same grouping
    test_df.loc[test_df.EDUCATION == 6, 'EDUCATION'] = 5

    X_test, y_test = test_df.drop(columns=['target']), test_df['target']

    with open(best_model_from, 'rb') as f:
        random_search_lgbm = pickle.load(f)

    from sklearn.metrics import f1_score
    y_pred = random_search_lgbm.best_estimator_.predict(X_test)
    f1_score(y_test, y_pred)
    
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    classification_report(y_test, y_pred, target_names=["Default payment: No", "Default payment: Yes"])
    
    ConfusionMatrixDisplay.from_estimator(
        random_search_lgbm.best_estimator_,
        X_test,
        y_test,
        display_labels=["Default payment: No", "Default payment: Yes"],
        values_format="d",
        cmap=plt.cm.Oranges,
    )
    
    # summary = pd.DataFrame(results_dict).T
    # summary['comments'] = [
    #     'Baseline model that just predict the most frequent class',
    #     'Simple logistic regression that performed better than the baseline',
    #     'Tuned logistic regression did not improve much, it took longer time to fit',
    #     'SVC performed worse than logistic regression, it has the longest fitting time',
    #     'Decision Tree has the poorest performance and greatly overfitted',
    #     'LGBM performed decently well',
    #     'XGBoost has similar performance as LGBM',
    #     'Random Forest was greatly overfitted',
    #     'LGBM with recursive feature elimination has negligible reduce in performance but it is simpler and has less overfitting',
    #     'LGBM with backward selection is similar to recursive feature elimination but has longer fitting time',
    #     'XGBoost with recursive feature elimination has negligible reduce in performance but it is simpler and has less overfitting',
    #     'XGBoost with backward selection is similar to recursive feature elimination but has longer fitting time',
    #     'Tuned LGBM with recursive feature elimination has the best performance',
    #     'Tuned XGBoost with recursive feature elimination improves in performance and also reduced in overfitting',
    # ]

if __name__ == '__main__':
    main()