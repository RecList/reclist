import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
from reclist.abstractions import RecModel

#TODO: Add methods for sequential problem
class RandomForest(RecModel):
    def __init__(self, regression_problem:bool = True, sequential_problem:bool=False, **kwargs):
        super().__init__()
        self.name = 'RandomForest'
        self.is_train = False
        available_models = {
            True:{'model': sklearn.ensemble.RandomForestRegressor, 'name': 'RandomForestRegressor'},
            False:{'model': sklearn.ensemble.RandomForestClassifier, 'name': 'RandomForestClassifier'}
        }
       # set model & name based on regression_problem:
        self._model = available_models[regression_problem]['model']
        self.name += available_models[regression_problem]['name']

        # set model params based on kwargs if provided:
        if 'model_config' in kwargs:
            self._model = self.model(kwargs['model_config'])
        else:
            self._model = self.model()

    
    
    def train(self, X_train:pd.DataFrame, y_train:pd.DataFrame):
        """
        Trains the model on the training data.
        
        Args:
            X_train (pd.DataFrame): Training data.
            y_train (pd.DataFrame): Training labels
        """
        self.is_train = True
        print('training random model...')
        self._model.fit(X_train, y_train.values.ravel())
    
    def candidate_generation(self, X_train:pd.DataFrame, y_train:pd.DataFrame, y_test,k:int=10):
        """
        Generates candidates based on the defined criteria. Implemneted in case of sequential problem.
        This method is then called to predict.
        Args:
            X_train (pd.DataFrame): Training data.
            y_train (pd.DataFrame): Training labels
            k (int, optional): Number of candidates to generate. Defaults to 10.
        Returns:
            pd.DataFrame: Candidate generation results.
        """
        pass

    def predict(self, test_data:pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions on test data.
        Args:
            test_data (pd.DataFrame): Test data to make predictions on.
            if sequential problem, test_data can be a list of lists. of pd.DataFrames of lists.
            in that case, explode the list of lists into a pd.DataFrame of lists 

        Raises:
            Exception: model_name not trained yet.

        Returns:
            pd.DataFrame: Predictions. if sequential problem, it can be pd.DataFrame of lists.
        """
        assert self.is_train, f'{self.name} not trained yet'

        preds= self._model.predict(test_data)
        preds= pd.DataFrame(preds)         
    
        return preds

    