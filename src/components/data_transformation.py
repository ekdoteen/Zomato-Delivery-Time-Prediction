import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle','Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 'distance_in_km']

            city_categories = ["Urban","Semi-Urban","Metropolitian"]
            festival_categories = ["No","Yes"]
            vehicle_categories = ["motorcycle","scooter","electric_scooter"]
            road_categories = ["Low","Medium","High","Jam"]
            weather_categories = ["Sunny","Windy","Cloudy","Sandstorms","Stormy","Fog"]

            # Create SimpleImputer object with median strategy for most columns
            numerical_imputer = SimpleImputer(strategy='median')

            # Create SimpleImputer object with most frequent strategy for 'multiple_deliveries' column
            multiple_deliveries_imputer = SimpleImputer(strategy='most_frequent')

            # Columns to be imputed with median strategy
            columns_to_impute_median = ['Delivery_person_Age','Delivery_person_Ratings','distance_in_km']

            # Columns to be imputed with most frequent strategy
            columns_to_impute_most_frequent = ['multiple_deliveries']

            logging.info("Pipeline initiated")
            # Define the numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('multiple_deliveries_imputer', multiple_deliveries_imputer),
                    ('numerical_imputer', numerical_imputer),
                    ('scaler', StandardScaler())
                    ]
            )

            cat_pipeline = Pipeline(

                steps = [
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ("ordnialencoder",OrdinalEncoder(categories=[weather_categories,road_categories,vehicle_categories,festival_categories,city_categories])),
                ("scaler",StandardScaler())    

                ]
            )
            
            preprocessor = ColumnTransformer(
                transformers=[
                ('num_pipeline',numerical_pipeline,columns_to_impute_most_frequent+columns_to_impute_median),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info("Pipeline completed")


        except Exception as e:
            logging.info("Error in Data Transformation")    
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            #reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train dataframe head : \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,"ID","Delivery_person_ID","Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude","Order_Date","Time_Orderd","Time_Order_picked","Type_of_order"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Transforming using preprocessor_obj

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and test datasets")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object (

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
                    

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")

            raise CustomException(e,sys)