# -Default-of-Credit-Card-Clients-Classification
This is a mini project to apply the concepts I have learned from DSCI571 and DSCI573.

From the root of the project run the following commands:

```
# Step 1: Prepare the dataset
python scripts/01_preprare_data.py \
    --input_data=data/raw/UCI_Credit_Card.csv \
    --write_to=data/processed/

# Step 2: Perform EDA
python scripts/02_eda.py \
    --input_data=data/processed/train_df.csv \
    --plot_to=results/figures/

# Step 3: Preprocessing
python scripts/03_preprocessing.py \
    --input_data=data/processed/train_df.csv \
    --preprocessor_to=results/models/ \
    --write_to=data/processed/

# Step 4: Fitting the models
python scripts/04_fit_classifier.py \
    --input_data=data/processed/train_df.csv \
    --preprocessor_from=results/models/preprocessor.pickle \
    --write_to=results/tables/ --best_model_to results/models/

# Step 5: Evaluate the model on test set
python scripts/05_evaluate_classifier.py \
    --input_data=data/processed/test_df.csv \
    --write_to=results/tables/ \
    --best_model_from=results/models/best_model.pickle 
