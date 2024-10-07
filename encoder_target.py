import os 
import pandas as pd

from utilsMsc.MyPreprocessing import PreprocessingDatasets

diretorio = "./Attacks"
le = PreprocessingDatasets()

for model_impt in ["gain",
                    "knn",
                    "mice",
                    "softimpute"]:
    
    for mecanismo in ["MAR-correlated_Multivariado",
                    "MNAR-determisticTrue_Multivariado"]:

        for missing_rate in [5,20,40]:
            complete_path = os.path.join(diretorio, "evasion",model_impt,"DatasetsUnificados",mecanismo)
            arq = f"liver_{model_impt}_md{missing_rate}.csv"
            df = pd.read_csv(os.path.join(complete_path, arq))
            df = le.label_encoder(df, ["target"])
            df.to_csv(os.path.join(complete_path,arq))