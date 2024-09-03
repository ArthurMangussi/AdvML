import xgboost as xgb
from MyADML import AdversarialML
from utilsMsc.MeLogSingle import MeLogger
from utilsMsc.MyPreprocessing import PreprocessingDatasets
from sklearn.model_selection import StratifiedKFold
from utilsMsc.MyUtils import MyPipeline
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score

def pipeline_baseline_classification_performance(tabela_resultados:dict):
    """
    Main Pipeline to perform the classification task across the baseline datasets 
    """
    _logger = MeLogger()
    try:
        classification_metrics = {"F1-score":[],
                                "Accuracy":[],
                                "Recall":[],
                                "AUC":[]}

        for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
            df = dados.copy()
            cv = StratifiedKFold()
            model = xgb.XGBClassifier()
            X = df.drop(columns = 'target')
            y = df['target'].values
            x_cv = X.values

            fold = 0
            for train_index, test_index in cv.split(x_cv, y):
                x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                y_treino, y_teste = y[train_index], y[test_index]

                X_treino = pd.DataFrame(x_treino, columns=X.columns)                    
                X_teste = pd.DataFrame(x_teste, columns=X.columns) 

                # Inicializando o normalizador (scaler)
                scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                # Normalizando os dados
                X_treino_norm = PreprocessingDatasets.normaliza_dados(scaler, X_treino)
                X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

                model.fit(X_treino_norm, y_treino)

                y_pred = model.predict(X_teste_norm)

                if len(np.unique(y)) == 2:
                    _logger.info(f"Dataset {nome} is binary -> {np.unique(y)}")
                    f1 = f1_score(y_true=y_teste, y_pred=y_pred)
                    acc = accuracy_score(y_true=y_teste, y_pred=y_pred)
                    rec = recall_score(y_true=y_teste, y_pred=y_pred)
                    auc = roc_auc_score(y_true=y_teste, y_score=model.predict_proba(X_teste_norm)[:,1])
                else:
                    _logger.info(f"Dataset {nome} is Multiclass with {len(np.unique(y))} classes -> {np.unique(y)}")
                    f1 = f1_score(y_true=y_teste, y_pred=y_pred, average="micro")
                    acc = accuracy_score(y_true=y_teste, y_pred=y_pred)
                    rec = recall_score(y_true=y_teste, y_pred=y_pred, average="micro")
                    auc = roc_auc_score(y_true=y_teste, 
                                        y_score=model.predict_proba(X_teste_norm),
                                        multi_class="ovr")

                classification_metrics["F1-score"].append({f"{nome}_fold{fold}":round(f1,3)})
                classification_metrics["Accuracy"].append({f"{nome}_fold{fold}":round(acc,3)})
                classification_metrics["Recall"].append({f"{nome}_fold{fold}":round(rec,3)})
                classification_metrics["AUC"].append({f"{nome}_fold{fold}":auc})
                fold += 1
                
            resultados = pd.DataFrame([classification_metrics])
            resultados.to_csv("./Baseline/classification_performance_baseline_datasets.csv")
        _logger.info("Resultados Baseline Classificação salvos com sucesso!")
    
    except Exception as erro:
        _logger.debug(f"Erro: {erro}")
    

if __name__ == "__main__":
    diretorio = "./DatasetsADVS"
    datasets = MyPipeline.carrega_datasets(diretorio)

    adv_ml = AdversarialML(datasets)
    tabela_resultados = adv_ml.cria_tabela()

    pipeline_baseline_classification_performance(tabela_resultados)
