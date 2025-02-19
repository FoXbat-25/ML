import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from exception import customException
from logger import logging
from utils import save_obj
import os

@dataclass
class DataTransftnConfg:
    preprocssr_obj_file_path=os.path.join('data','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_confg = DataTransftnConfg()
        except Exception as e:
            raise customException(e, sys)
        
    def get_data_transformer_obj(self):
        try:
            num_features=['reading score','writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(sparse=False)),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise customException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Initiated data transformation')
            preprocessing_obj=self.get_data_transformer_obj()
            target_column='math score'
            
            input_train_features=train_df.drop(columns=[target_column], axis=1)
            target_train_feature=train_df[target_column]

            input_test_features=test_df.drop(columns=[target_column], axis=1)
            target_test_feature=test_df[target_column]

            input_trained_arr=preprocessing_obj.fit_transform(input_train_features)
            input_test_arr=preprocessing_obj.transform(input_test_features)

            train_arr = np.c_[input_trained_arr, np.array(target_train_feature)]
            test_arr = np.c_[input_test_arr, np.array(target_test_feature)]

            save_obj(
                file_path=self.data_transformation_confg.preprocssr_obj_file_path, obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_confg.preprocssr_obj_file_path
            )

            
        except Exception as e:
            raise customException(e, sys)