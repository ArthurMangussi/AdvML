import pandas as pd
from utilsMsc.MeLogSingle import MeLogger
from utilsMsc.MyUtils import MyPipeline
from MyADML import AdversarialML
import os
from sdmetrics.single_column import KSComplement, CSTest


def pipeline_data_distribution_tests(attack:str,
                                    model_impt:str,
                                    md_mechanism:str,
                                                 ):
    """
    Function to calculate the Kolmogorov-Smirnov test
    """
    _logger = MeLogger()

    # Read the attacked datasets
    diretorio = "./Attacks"
    complete_path = os.path.join(diretorio, attack,model_impt,"DatasetsUnificados",md_mechanism)
    
    # Read the original datasets
    diretorio_original_data = "./DatasetsADVS"
    datasets = MyPipeline.carrega_datasets(diretorio_original_data)

    adv_ml = AdversarialML(datasets)
    tabela_resultados = adv_ml.cria_tabela()

    ks_test_all = {"Dataset":[],
                    "Missing rate":[],
                    "Column":[],
                    "ks_stat":[]
                    }
    cs_test_all = {"Dataset":[],
                    "Missing rate":[],
                    "Column":[],
                    "cs_stat":[]
                    }

                                        
    for dados, name in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
        df_original = dados.copy()
        binary_features = MyPipeline.get_binary_features(data=df_original)

        try: 
            for missing_rate in [5,20,40]:
                df_path = complete_path + f"\\{name}_{model_impt}_md{missing_rate}.csv"
                dados_imputed = pd.read_csv(df_path)
                df_imputed = dados_imputed.copy()
        
                # Calculate the Kolmogorov-Smirnov for numerical features and 
                for col in df_original.columns:
                    if col not in binary_features:
                        test_stat = KSComplement.compute(df_original[col].values, df_imputed[col].values)
                        ks_test_all["Dataset"].append(name)
                        ks_test_all["Column"].append(col)
                        ks_test_all["Missing rate"].append(missing_rate)
                        ks_test_all["ks_stat"].append(test_stat)

                    else:
                        test_stat = CSTest.compute(df_original[col].values, df_imputed[col].values)
                        cs_test_all["Dataset"].append(name)
                        cs_test_all["Column"].append(col)
                        cs_test_all["Missing rate"].append(missing_rate)
                        cs_test_all["cs_stat"].append(test_stat)
                    
                    

        except Exception as erro:
            _logger.debug(f"Erro: {erro}")

    resultados_ks = pd.DataFrame(ks_test_all)
    resultados_ks.to_csv(f"./Attacks/{attack}/{model_impt}/Resultados/ks_test_{attack}_{model_impt}_{mecanismo}.csv", index=False)
    resultados_cs = pd.DataFrame(cs_test_all)
    resultados_cs.to_csv(f"./Attacks/{attack}/{model_impt}/Resultados/cs_test_{attack}_{model_impt}_{mecanismo}.csv", index=False)
    return _logger.info("Resultados salvos com sucesso!")
    


if __name__ == "__main__":
    for mecanismo in ["MAR-correlated_Multivariado", "MNAR-determisticTrue_Multivariado"]:
            for attack_str in ["evasion","poison"]:
        
                pipeline_data_distribution_tests(attack=attack_str,
                                                            model_impt="mice",
                                                            md_mechanism=mecanismo,
                                                            )
                pipeline_data_distribution_tests(attack=attack_str,
                                                            model_impt="knn",
                                                            md_mechanism=mecanismo,
                                                            )
                pipeline_data_distribution_tests(attack=attack_str,
                                                            model_impt="gain",
                                                            md_mechanism=mecanismo,
                                                            )
                pipeline_data_distribution_tests(attack=attack_str,
                                                            model_impt="softimpute",
                                                            md_mechanism=mecanismo,
                                                            )
    