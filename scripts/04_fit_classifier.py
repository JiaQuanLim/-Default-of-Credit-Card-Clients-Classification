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
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate


@click.command()
@click.option('--input_data', type = str, help = "Path to processed training data")
#@click.option('--plot_to', type = str, help = "Path to directory where the plot will be written to")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
@click.option('--best_model_to', type = str, help = "Path to directory where the best_model will be written to")
@click.option('--preprocessor_from', type=str, help="Path to directory where the preprocessor object lives")


def main(input_data, preprocessor_from, write_to, best_model_to):
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

    with open(preprocessor_from, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Combining education group 6 to group 5 as both belongs to the same grouping
    train_df.loc[train_df.EDUCATION == 6, 'EDUCATION'] = 5
    X_train, y_train = train_df.drop(columns=['target']), train_df['target']

    # Code adapted from DSCI571: Lecture 4 
    def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
        """
        Returns mean and std of cross validation
    
        Parameters
        ----------
        model :
            scikit-learn model
        X_train : numpy array or pandas DataFrame
            X in the training data
        y_train :
            y in the training data
    
        Returns
        ----------
            pandas Series with mean scores from cross_validation
        """
    
        scores = cross_validate(model, X_train, y_train, **kwargs)
    
        mean_scores = pd.DataFrame(scores).mean()
        std_scores = pd.DataFrame(scores).std()
        out_col = []
    
        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))
    
        return pd.Series(data=out_col, index=mean_scores.index)
    
    dc = DummyClassifier(strategy='stratified', random_state=123)
    
    results_dict = {}
    results_dict["dummy"] = mean_std_cross_val_scores(dc, X_train, y_train, scoring='f1', return_train_score=True)
    pd.DataFrame(results_dict)

    # 1. Linear model
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    
    pipe_lr = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, class_weight='balanced', random_state=123))
    
    # 2. Hyperparameter tuning
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform, randint
    
    param_dist = {"logisticregression__C": loguniform(1e-3, 1e3)}
    
    logreg_tuned = RandomizedSearchCV(pipe_lr, param_dist, n_iter=100, scoring='f1', random_state=123, n_jobs=-1)
    logreg_tuned.fit(X_train, y_train)
    
    results_dict["Logistic Regression"] = mean_std_cross_val_scores(pipe_lr, X_train, y_train, scoring='f1', return_train_score=True)
    results_dict["Logistic Regression (Tuned)"] = mean_std_cross_val_scores(logreg_tuned, X_train, y_train, scoring='f1', return_train_score=True)
    
    
    # ==========================================================================================
    from sklearn.svm import SVC
    from lightgbm.sklearn import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    models = {
            "SVC": SVC(random_state=123),
            "Decision Tree": DecisionTreeClassifier(random_state=123),
            "LGBM": LGBMClassifier(is_unbalance=True, force_row_wise=True, random_state=123),
            "XGBoost": XGBClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], 
                                     random_state=123, 
                                     verbosity=0),
            "Random Forest": RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=123)
        }
    
    for model_name, model in models.items():
            clf_pipe = make_pipeline(preprocessor, model)
            results_dict[model_name] = mean_std_cross_val_scores(clf_pipe, X_train, y_train, scoring='f1', return_train_score=True)
    
    # ==========================================================================================
    from sklearn.feature_selection import RFECV
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import Ridge
    
    feature_selection = {
            "rfe": RFECV(Ridge()),
            "backward": SequentialFeatureSelector(Ridge(), direction="backward")}
    
    for fs_name, fs in feature_selection.items():
            fs_pipe = make_pipeline(preprocessor, 
                                    fs, 
                                    LGBMClassifier(is_unbalance=True, force_row_wise=True, random_state=123))
            results_dict['LGBM_'+fs_name] = mean_std_cross_val_scores(fs_pipe, X_train, y_train, scoring='f1', return_train_score=True)
    
    for fs_name, fs in feature_selection.items():
            fs_pipe = make_pipeline(preprocessor, 
                                    fs, 
                                    XGBClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], 
                                     random_state=123, 
                                     verbosity=0))
            results_dict['XGBoost_'+fs_name] = mean_std_cross_val_scores(fs_pipe, X_train, y_train, scoring='f1', return_train_score=True)
    
    for fs_name, fs in feature_selection.items():
            fs_pipe = make_pipeline(preprocessor, 
                                    fs, 
                                    XGBClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], 
                                     random_state=123, 
                                     verbosity=0))
            results_dict['XGBoost_'+fs_name] = mean_std_cross_val_scores(fs_pipe, X_train, y_train, scoring='f1', return_train_score=True)
    
    #==========================================================================================
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import expon, lognorm, loguniform, randint, uniform, norm, randint
    
    param_dist = {
        'lgbmclassifier__learning_rate': uniform(0.01, 0.1),
        'lgbmclassifier__subsample': uniform(0, 1),
        'lgbmclassifier__max_depth': randint(3, 10),
        'lgbmclassifier__num_leaves': randint(15, 30)
    }
    
    lgbm_pipe = make_pipeline(preprocessor, 
                             RFECV(Ridge()),
                             LGBMClassifier(is_unbalance=True, verbose=-1, random_state=123))
    
    random_search_lgbm = RandomizedSearchCV(
        lgbm_pipe,
        param_dist,
        n_iter=20,  
        verbose=1,
        scoring="f1",
        random_state=123,
        return_train_score=True,
        n_jobs=-1
    )
    random_search_lgbm.fit(X_train, y_train)
    
    results_dict['LGBM_rfe (Tuned)'] = mean_std_cross_val_scores(random_search_lgbm.best_estimator_, 
                                                                X_train, 
                                                                y_train, 
                                                                scoring='f1', 
                                                                return_train_score=True)
    
    param_dist = {
        'xgbclassifier__max_depth': randint(3, 10),
        'xgbclassifier__learning_rate': uniform(0.01, 0.1),
        'xgbclassifier__subsample': uniform(0, 1),
        'xgbclassifier__n_estimators':randint(50, 200)
    }
    
    xgb_pipe = make_pipeline(preprocessor, 
                             RFECV(Ridge()),
                             XGBClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], 
                                           random_state=123,
                                           verbosity=0))
    
    random_search_xgb = RandomizedSearchCV(
        xgb_pipe,
        param_dist,
        n_iter=20,  
        verbose=1,
        scoring="f1",
        random_state=123,
        return_train_score=True,
        n_jobs=-1
    )
    random_search_xgb.fit(X_train, y_train)
    
    results_dict['XGBoost_rfe (Tuned)'] = mean_std_cross_val_scores(random_search_xgb.best_estimator_, 
                                                                X_train, 
                                                                y_train, 
                                                                scoring='f1', 
                                                                return_train_score=True)
    pd.DataFrame(results_dict).T.to_csv(os.path.join(write_to, "cross_validation.csv"), index=True)

    with open(os.path.join(best_model_to, "best_model.pickle"), 'wb') as f:
        pickle.dump(random_search_lgbm, f)

if __name__ == '__main__':
    main()