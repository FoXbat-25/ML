import os
import sys
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from exception import customException
from logger import logging
from utils import save_obj, model_eval
from dataclasses import dataclass

@dataclass
class modelTrainerConfg:
    trained_model_file_path=os.path.join('data', 'model.pkl')
class modelTrainer:
    def __init__(self):
        self.model_trainer_confg=modelTrainerConfg()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            x_train, y_train, x_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "random_forest":RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "linear_regression": LinearRegression(),
                "knn": KNeighborsRegressor(),
                "xgboost":XGBRegressor(),
                "catboost":CatBoostRegressor(verbose=False),
                "adaboost":AdaBoostRegressor()
            }
            
            model_report:dict=model_eval(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            print(model_report)
            
            
        except Exception as e:
            raise customException(e, sys)
            

