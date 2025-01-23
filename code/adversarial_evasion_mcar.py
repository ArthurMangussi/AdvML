import pandas as pd 

from sklearn.model_selection import StratifiedKFold
from utilsMsc.MyModels import ModelsImputation
from utilsMsc.MyUtils import MyPipeline
from utilsMsc.MyPreprocessing import PreprocessingDatasets
from utilsMsc.MeLogSingle import MeLogger
from utilsMsc.MyResults import AnalysisResults

from utilsMsc.MyADML import AdversarialML
import multiprocessing

from mdatagen.multivariate.mMCAR import mMCAR

from time import perf_counter
import os 

def pipeline_adversarial(model_impt:str, mecanismo:str, tabela_resultados:dict,set_attack:str):
    "Main pipeline to perform adversarial evasion attack under MAR multivariate mechanism."
    _logger = MeLogger()

    # Cria diretórios para salvar os resultados do experimento
    os.makedirs(f"./Attacks/{set_attack}/{model_impt}/Tempos/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Attacks/{set_attack}/{model_impt}/Datasets/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Attacks/{set_attack}/{model_impt}/Resultados/{mecanismo}_Multivariado", exist_ok=True)
    
    with open(f'./Attacks/{set_attack}/{model_impt}/Tempos/{mecanismo}_Multivariado/tempo_{model_impt}.txt','w') as file:
        for dados, nome in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
            df = dados.copy()
            X = df.drop(columns='target')
            y = df['target'].values
            binary_features = MyPipeline.get_binary_features(data=df)
            for md in tabela_resultados['missing_rate']:
                file.write(f'Dataset = {nome} com MD = {md}\n')
                _logger.info(f'Dataset = {nome} com MD = {md} no {model_impt}\n')

                fold = 0
                cv = StratifiedKFold()
                x_cv = X.values

                for train_index, test_index in cv.split(x_cv, y):
                    _logger.info(f'Fold = {fold}')
                    x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                    y_treino, y_teste = y[train_index], y[test_index]
                
                    X_treino = pd.DataFrame(x_treino, columns=X.columns)
                    X_teste = pd.DataFrame(x_teste, columns=X.columns)

                    # Geração do ataque no dataset de teste
                    X_adv_test = AdversarialML.attack_evasion(X_train=X_treino,
                                                          y_train=y_treino,
                                                          X_test=X_teste,
                                                          y_test=y_teste)

                    # Inicializando o normalizador (scaler)
                    scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                    # Normalizando os dados
                    X_treino_norm = PreprocessingDatasets.normaliza_dados(
                        scaler, X_treino
                    )
                    X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_adv_test)

                    # Geração dos missing values em cada conjunto de forma independente
                    impt_md_train = mMAR(X=X_treino_norm, 
                                            y=y_treino, 
                                            )
                    X_treino_norm_md = impt_md_train.correlated(
                        missing_rate=md
                    )
                    X_treino_norm_md = X_treino_norm_md.drop(columns='target')

                    impt_md_test = mMAR(X=X_teste_norm, 
                                         y=y_teste,
                                        )
                    X_teste_norm_md = impt_md_test.correlated(
                        missing_rate=md
                    )
                    X_teste_norm_md = X_teste_norm_md.drop(columns='target')
                    
                    inicio_imputation = perf_counter()
                    # Inicializando e treinando o modelo
                    model_selected = ModelsImputation()
                    if model_impt == 'saei':
                        # SAEI
                        model = model_selected.choose_model(
                            model=model_impt,
                            x_train=X_treino_norm,
                            x_test=X_teste_norm,
                            x_train_md=X_treino_norm_md,
                            x_test_md=X_teste_norm_md,
                            input_shape=X.shape[1],
                        )

                    # KNN, MICE, PMIVAE, MEAN, SoftImpute, GAIN, missForest
                    else:
                        model = model_selected.choose_model(
                            model=model_impt,
                            x_train=X_treino_norm_md,
                            x_test = X_teste_norm_md,
                            x_test_complete = X_teste_norm,
                            binary_val = binary_features                         
                        )

                    fim_imputation = perf_counter()
                    file.write(
                        f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.3f} s\n'
                    )

                    # Imputação dos missing values nos conjuntos de treino e teste
                    try:
                        if model_impt == "mean":
                            output_md_test = model.transform(
                                X_teste_norm_md
                            )                        
                        else:
                            output_md_test = model.transform(
                                X_teste_norm_md.iloc[:, :].values
                            )
                    except AttributeError:                        
                        fatores_latentes_test = model.fit(X_teste_norm_md.iloc[:, :].values)
                        output_md_test = model.predict(X_teste_norm_md.iloc[:, :].values)

                    # Encode das variávies binárias
                    df_output_md_teste = pd.DataFrame(output_md_test, columns=X.columns)
                    output_md_test = MyPipeline.encode_features_categoricas(list_binary_features=binary_features,
                                                                            imputed_dataset=df_output_md_teste)

                    # Calculando MAE para a imputação no conjunto de teste
                    (
                        mae_teste_mean,
                        mae_teste_std,
                    ) = AnalysisResults.gera_resultado_multiva(
                        resposta=output_md_test,
                        dataset_normalizado_md=X_teste_norm_md,
                        dataset_normalizado_original=X_teste_norm,
                    )

                    tabela_resultados[
                        f'{model_impt}/{nome}/{md}/{fold}/MAE'
                    ] = {'teste': round(mae_teste_mean,3)}
                    
                    # Dataset imputado
                    data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                    data_imputed['target'] = y_teste

                    data_imputed.to_csv(f"./Attacks/{set_attack}/{model_impt}/Datasets/{mecanismo}_Multivariado/{nome}_{model_impt}_fold{fold}_md{md}.csv", index=False)
                    fold += 1
                    
            resultados_final = AnalysisResults.extrai_resultados(tabela_resultados)

            # Resultados da imputação: salvos por dataset para ganhar tempo de processamento
            resultados_mecanismo = (
                AnalysisResults.calcula_metricas_estatisticas_resultados(
                    resultados_final, 1, fold
                )
            )
            resultados_mecanismo.to_csv(
                f'./Attacks/{set_attack}/{model_impt}/Resultados/{mecanismo}_Multivariado/{nome}_{model_impt}_{mecanismo}.csv',
                
            )        
    return _logger.info(f"Imputation_{model_impt}_done!")

if __name__ == "__main__":

    diretorio = "./data"
    datasets = MyPipeline.carrega_datasets(diretorio)

    adv_ml = AdversarialML(datasets)
    tabela_resultados = adv_ml.cria_tabela()
    
    attack_str = "evasion"
    mecanismo = "MAR-correlated"
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

        args_list = [
                     ("knn",mecanismo,tabela_resultados,attack_str),
                     ("mice",mecanismo,tabela_resultados,attack_str),
                     ("softImpute",mecanismo,tabela_resultados,attack_str),
                     ("gain",mecanismo,tabela_resultados,attack_str)
                     ]
        
        pool.starmap(pipeline_adversarial,args_list)

