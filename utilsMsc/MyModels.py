# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


# MICE, KNN, Dumb, missForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

# Soft impute 
from Algorithms.soft_impute import SoftImpute

# Generative Adversarial Imputation Networks
from Algorithms.gain import Gain

import pandas as pd
import numpy as np
import warnings

from utilsMsc.MeLogSingle import MeLogger

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

class ModelsImputation:
    def __init__(self) :
        self._logger = MeLogger()
    # ------------------------------------------------------------------------
    @staticmethod
    def model_mice(dataset_train:pd.DataFrame):
        imputer = IterativeImputer(max_iter=100)
        mice = imputer.fit(dataset_train.iloc[:, :].values)

        return mice

    # ------------------------------------------------------------------------
    @staticmethod
    def model_knn(dataset_train:pd.DataFrame):
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(dataset_train.iloc[:, :].values)

        return knn

    # ------------------------------------------------------------------------
    @staticmethod
    def model_dumb(dataset_train:pd.DataFrame, 
                   binary_vals:list[str]):
        
        numeric_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        num_vals = [col for col in dataset_train.columns if col not in binary_vals]
        try:
            copy_binary_vals = binary_vals.copy()
            copy_binary_vals.remove("target")
        except ValueError:
            print(binary_vals)
            
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_imputer, num_vals),
                ('cat', categorical_imputer, copy_binary_vals)
            ])

        dumb = preprocessor.fit(dataset_train)
        return dumb
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_softimpute(dataset_train:pd.DataFrame):
        imputer = SoftImpute()
        soft_impute = imputer.fit(dataset_train.iloc[:,:].values)
        return soft_impute

    # ------------------------------------------------------------------------
    @staticmethod
    def model_gain(dataset_train:pd.DataFrame):
        imputer = Gain(batch_size=32,
                       iterations=10000)
        gain = imputer.fit(dataset_train.iloc[:,:].values)
        return gain
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_missForest(dataset_train:pd.DataFrame):
        rf = RandomForestRegressor(n_jobs=-1, 
                                   criterion="absolute_error", 
                                   n_estimators=10, 
                                   random_state=42)
        imputer = IterativeImputer(estimator=rf)
        missForest = imputer.fit(dataset_train.iloc[:, :].values)

        return missForest
        
    # ------------------------------------------------------------------------
    def choose_model(self,model: str, x_train, **kwargs):
        match model:
            case "mice":
                self._logger.info("[MICE] Training...")
                return ModelsImputation.model_mice(x_train)

            case "knn":
                self._logger.info("[KNN] Training...")
                return ModelsImputation.model_knn(x_train)

            case "pmivae":
                self._logger.info("[PMIVAE] GridSearch...")
                params = {
                        "epochs": [200],
                        "latent_dimension": [5,10],
                        "neurons": [[np.shape(x_train)[0]/2],
                                    [np.shape(x_train)[0]/2, np.shape(x_train)[0]/4]],
                    }
                best_params, best_score = ModelsImputation.GridSearchPMIVAE(X_train=x_train,
                                                                            X_test=kwargs["x_test"],
                                                                            param_grid=params,
                                                                            X_test_complete=kwargs["x_test_complete"])
                
                self._logger.info(f"Best params for PMIVAE: {best_params}")
                self._logger.info(f"Best score found in GridSearch (MSE): {best_score}")
                
                return ModelsImputation.model_autoencoder_pmivae(x_train.loc[:, :].values, 
                                                                 params=best_params)

            case "saei":
                self._logger.info("[SAEI] Training...")
                return ModelsImputation.modelo_saei(
                    dataset_train=x_train,
                    dataset_test=kwargs["x_test"],
                    dataset_train_md=kwargs["x_train_md"],
                    dataset_test_md=kwargs["x_test_md"],
                    input_shape=kwargs["input_shape"],
                )

            case "mean":
                self._logger.info("[MEAN] Training...")
                return ModelsImputation.model_dumb(x_train, 
                                                   kwargs["binary_val"])
            
            case "softImpute":
                self._logger.info("[SoftImpute] Training...")
                return ModelsImputation.model_softimpute(x_train)
            
            case "gain":
                self._logger.info("[GAIN] Training...")
                return ModelsImputation.model_gain(x_train)
            
            case "missForest":
                self._logger.info("[missForest] Training...")
                return ModelsImputation.model_missForest(x_train)

