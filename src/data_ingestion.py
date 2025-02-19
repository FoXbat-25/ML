import os
import sys
from datetime import datetime
from exception import customException
from logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfg:
    train_data_path:str = os.path.join('data', 'train_data.csv')
    test_data_path:str = os.path.join('data', 'test_data.csv')
    raw_data_path:str = os.path.join('data', 'data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_confg = DataIngestionConfg()
    def ingestion(self):
        logging.info(f"Initialising data ingestion")
        try:
            df = pd.read_csv(r'C:\Users\dhruv\Downloads\Projects\ML\exams.csv')
            os.makedirs(os.path.dirname(self.ingestion_confg.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_confg.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_confg.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_confg.test_data_path, index=False, header=True)
            logging.info('Data split complete')
            return(
                self.ingestion_confg.train_data_path,
                self.ingestion_confg.test_data_path

            )
        except Exception as e:
            raise customException(e, sys)

if __name__ == "__main__":
    obj=DataIngestion()
    obj.ingestion()