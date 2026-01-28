import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.FSAAZ.exception import CustomException
from src.FSAAZ.logger import logging
from src.FSAAZ.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    le_obj_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder = LabelEncoder()
        self.preprocessor = None

    def get_data_transformation_object(self, train_df):
        try:
            # Financial sentiment analysis features (update based on your dataset)
            numerical_columns = ['length', 'word_count', 'avg_word_len', 'punctuation_count', 
                               'upper_case_count', 'sentiment_score', 'subjectivity']
            
            categorical_columns = ['source', 'market_type']  # e.g., 'twitter', 'news', 'reddit'
            
            # Numerical pipeline: Handle missing + scale
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Categorical pipeline: Handle missing + one-hot encode
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # Combine both pipelines
            self.preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return self.preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Reading train and test files completed')
            logging.info(f'Train data shape: {train_df.shape}')
            logging.info(f'Test data shape: {test_df.shape}')

            # Get preprocessor (fits on train data)
            preprocessing_obj = self.get_data_transformation_object(train_df)
            
            target_column_name = 'Sentiment'  # Target: positive/neutral/negative
            
            # Prepare features (drop target only, keep all feature columns)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Training features shape: {input_feature_train_df.shape}")
            
            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Encode labels (fit on train only)
            target_feature_train_encoded = self.label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = self.label_encoder.transform(target_feature_test_df)
            
            # Combine features + targets
            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]
            
            # Save both preprocessor and label encoder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path=self.data_transformation_config.le_obj_file_path,
                obj=self.label_encoder
            )
            
            logging.info("✅ Data transformation completed and objects saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.le_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
