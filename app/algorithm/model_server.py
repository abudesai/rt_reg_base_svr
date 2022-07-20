import numpy as np
import os

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.regressor as regressor


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path): 
        self.model_path = model_path
        
    
    def _get_preprocessor(self): 
        try: 
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
            return self.preprocessor
        except: 
            print(f'No preprocessor found to load from {self.model_path}. Did you train the model first?')
        return None
    
    
    def _get_model(self): 
        try: 
            self.model = regressor.load_model(self.model_path)
            return self.model
        except: 
            print(f'No model found to load from {self.model_path}. Did you train the model first?')
        return None
    
        
    
    def predict(self, data, data_schema):  
        
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)        
        # make predictions
        preds = model.predict( pred_X )
        # inverse transform the predictions to original scale
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)        
        # get the names for the id and prediction fields
        id_field_name = data_schema["inputDatasets"]["regressionBaseMainInput"]["idField"]     
        # return te prediction df with the id and prediction fields
        preds_df = data[[id_field_name]].copy()
        preds_df['prediction'] = np.round(preds,4)
        
        return preds_df
        
        

