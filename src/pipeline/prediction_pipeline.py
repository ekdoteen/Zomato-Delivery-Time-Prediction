import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Vehicle_condition:float,
                 multiple_deliveries:float,
                 distance_in_km:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Type_of_vehicle:str,
                 Festival:str,
                 City:str):
        
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Vehicle_condition=Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.distance_in_km = distance_in_km
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City
    

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Vehicle_condition':[self.Vehicle_condition],
                'multiple_deliveries':[self.multiple_deliveries],
                'distance_in_km':[self.distance_in_km],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'Festival':[self.Festival],
                'City':[self.City]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
        


'''if __name__=='__main__':
    data=CustomData(
                 Delivery_person_Age=36.0,
                 Delivery_person_Ratings=4.2,
                 Vehicle_condition=10.27,
                 multiple_deliveries=1,
                 distance_in_km=10.0,
                 Weather_conditions="Sunny",
                 Road_traffic_density="Jam",
                 Type_of_vehicle="motorcycle",
                 Festival='No',
                 City='Metropolitian'
            )
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)

    results=round(pred[0],2)
    print(results)'''