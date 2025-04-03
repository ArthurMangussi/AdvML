if __name__ == "__main__":

    type_attack = "Carlini"
    
    for impt in ["gain", "knn", "mice", "softImpute"]:
        for mechanism in ["MAR-correlated-UCI_Multivariado",
                "MNAR-determisticTrue-UCI_Multivariado",
                "MCAR_Multivariado"
                ]:
            for dataset in ["acute-inflammations",
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
                    path = f'./Complexidade/{type_attack}/{mechanism}/{impt}/{dataset}_md{mr}.arff'

                    with open(path) as f:
                        data = f.readlines()

                    if dataset == "hcv-egyptian":
                        data[30] = data[30].replace("{1.0, 2.0, 3.0, 4.0}", "{0.0, 1.0, 2.0, 3.0}")
                    elif dataset == "mathernal-risk":
                        data[8] = data[8].replace("{0, 1, 2}", "{0.0, 1.0, 2.0}")
                    elif dataset == "npha":
                        data[15] = data[15].replace("{0, 1, 2}", "{0.0, 1.0, 2.0}")
                    elif dataset == "contraceptive-methods":
                        data[11] = data[11].replace("{0, 1, 2}", "{0.0, 1.0, 2.0}")
                        data[9] = data[9].replace("@ATTRIBUTE husband_ocupation NUMERIC", "@ATTRIBUTE husband_ocupation1 NUMERIC")
                    elif dataset == "iris":
                        data[6] = data[6].replace("{0, 1, 2}", "{0.0, 1.0, 2.0}")
                    elif dataset == "wine":
                        data[15] = data[15].replace("{0, 1, 2}", "{0.0, 1.0, 2.0}")

                    with open(path, "w") as f:
                        f.writelines(data)