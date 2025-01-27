import pandas as pd 

from sklearn.model_selection import StratifiedKFold
from utilsMsc.MeLogSingle import MeLogger

from utilsMsc.MyADML import AdversarialML
import os

from utilsMsc.MyPreprocessing import PreprocessingDatasets

def pipeline_save_models(tabela_resultados:dict):
    
    _logger = MeLogger()
    os.makedirs("./models/", exist_ok=True)
    
    for dados, nome in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
        df = dados.copy()
        X = df.drop(columns='target')
        y = df['target'].values

        fold = 0
        cv = StratifiedKFold()
        x_cv = X.values

        for train_index, test_index in cv.split(x_cv, y):
            x_treino, x_teste = x_cv[train_index], x_cv[test_index]
            y_treino, y_teste = y[train_index], y[test_index]
        
            X_treino = pd.DataFrame(x_treino, columns=X.columns)

            # Inicializando o normalizador (scaler)
            scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

            # Normalizando os dados
            X_treino_norm = PreprocessingDatasets.normaliza_dados(
                scaler, X_treino
            )

            AdversarialML.train_and_save(X_treino_norm,y_treino,fold)
            _logger.info(f'SVM trained on Fold = {fold}')