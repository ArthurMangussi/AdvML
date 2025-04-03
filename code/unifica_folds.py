import os 
import pandas as pd


for type_attack in ["Carlini"]:

  for model_impt in ["gain",
                        "knn",
                        "mice",
                        "softImpute"]:
      for mecanismo in [#"MAR-correlated-UCI_Multivariado",
                        "MNAR-determisticTrue-UCI_Multivariado",
                        #"MCAR_Multivariado"
                        ]:
        
        os.makedirs(f"./Attacks/Resultados_UCI/{type_attack}/{model_impt}/DatasetsUnificados/{mecanismo}", exist_ok=True)

        for name_dataset in ["acute-inflammations",
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
                  #"phoneme",
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

              folds = []
              for fold in range (5):
                
                path = f"./Attacks/Resultados_UCI/{type_attack}/{model_impt}/Datasets/{mecanismo}"
                arq = f"{name_dataset}_{model_impt}_fold{fold}_md{mr}.csv"
                    
                df = pd.read_csv(os.path.join(path,arq))
                folds.append(df)
                    
              df_unificado = pd.concat(folds, ignore_index=True)
              
              df_unificado.to_csv(f"./Attacks/Resultados_UCI/{type_attack}/{model_impt}/DatasetsUnificados/{mecanismo}/{name_dataset}_{model_impt}_md{mr}.csv", index=False)        

print("done")