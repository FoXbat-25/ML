A Simple ML project to predict math scores based on the other given parameters. This model makes use of Random Forest, Gradient Boosting, Decision Trees, Linear Regression , KNN, XGRegressor, CatBoost and AdaBoost regression models and compares their performance using R^2 score. 

The entire process flow goes like this:-

Step 1: Data Ingestion - From locally downloaded exams.csv
Step 2: Data transformation - Numerical features - Mean Imputers, Standardisation of values
                            - Categorical features - Imputers (Most frequent), Parsing (OHE), Standarisation

Step 3: Model training - Train test split (80-20), Model eval (R2 Score)
