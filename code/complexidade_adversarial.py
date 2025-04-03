# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


import pandas as pd
from utilsMsc.MyComplexity import ComplexityDatasets
from time import perf_counter

if __name__ == '__main__':

    init = perf_counter()
    type_attack = "FGSM"
    bs = {}

    for impt in ["gain", "knn", "mice", "softImpute"]:
        for mechanism in ["MAR-correlated_Multivariado",
                "MNAR-determisticTrue_Multivariado",
                "MCAR_Multivariado"
                ]:
            for name in ["acute-inflammations",
                  "autism-adolescent",
                  "autism-adult",
                  "autism-child",
                  "bc-coimbra",
                  "blood-transfusion",
                  "contraceptive-methods",
                  "diabetic",
                  "echocardiogram",
                  "fertility",
                  "german-credit",
                  "haberman",
                  "hcv-egyptian",
                  "heart-cleveland",
                  "hepatitis",
                  "iris",
                  "liver",                                              
                  "mathernal-risk",
                  "npha",
                  "parkinsons",
                  "phoneme",
                  "pima-diabetes",
                  "proba_football",
                  "ricci",
                  "sa-heart",
                  "thoracic-surgery",
                  "thyroid-recurrence",
                  "wine",
                  "wiscosin"
                        ]:
                for mr in [5,20,40]:
                    diretorio = f"./Attacks/Resultados_UCI/{type_attack}/{impt}/DatasetsUnificados/{mechanism}/{name}_{impt}_md{mr}.csv"
    
                    # ComplexityDatasets.cria_arquivo_arff(path=diretorio,
                    #                      type_attack=type_attack,
                    #                      mechanism=mechanism,
                    #                      model_impt=impt,
                    #                      nome_dataset=name,
                    #                      mr=mr)
                    
                    print(f"Complexidade --> {mechanism} - {name}_md{mr}")
                    path = f'./Complexidade/{type_attack}/{mechanism}/{impt}/{name}_md{mr}.arff'
                    bs[f"{impt}_{mechanism}_{name}_md{mr}"] = ComplexityDatasets.analisa_complexidade(path)

            pd.DataFrame(bs).T.to_excel(
                        f'./Complexidade/{type_attack}/{mechanism}/{impt}_{mechanism}_complexidade_results.xlsx'
                    )

fim = perf_counter()
print(f'Tempo de processamento: {fim-init:.4f}')