from sklearn.model_selection import RandomizedSearchCV
from utilsMsc.MyPreprocessing import PreprocessingDatasets
from utilsMsc.MeLogSingle import MeLogger

import numpy as np
import pandas as pd

from sklearn.svm import SVC
import joblib

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import CarliniL2Method, FastGradientMethod, ProjectedGradientDescent
from art.attacks.poisoning import PoisoningAttackSVM

class AdversarialML:
    def __init__(self, datasets:dict):
        self._logger = MeLogger()
        self._prep = PreprocessingDatasets()
        self.datasets = datasets
        self.wiscosin = self.pre_processing_wiscosin()
        self.pima = self.pre_processing_pima()
        self.acute = self.pre_processing_acute()
        self.autism_teen = self.pre_processing_autism_teen()
        self.autism_adult = self.pre_processing_autism_adult()
        self.autism_child = self.pre_processing_autism_child()
        self.bank = self.pre_processing_bank()
        self.blood_transfusion = self.pre_processing_blood()
        self.bc_coimbra = self.pre_processing_bcCoimbra()
        self.contraceptive = self.pre_processing_contraceptive()
        self.diabetic = self.pre_processing_diabetic()
        self.echocardiogram = self.pre_processing_echocardiogram()
        self.phoneme = self.pre_processing_phoneme()
        self.fertility = self.pre_processing_fertility()
        self.haberman = self.pre_processing_haberman()
        self.hcv = self.pre_processing_HCV()
        self.cleveland = self.pre_processing_cleveland()
        self.hepatitis = self.pre_processing_hepatitis()
        self.iris = self.pre_processing_iris()
        self.liver = self.pre_processing_indianLiver()
        self.thyroid = self.pre_processing_thyroid()
        self.mathernal_risk = self.pre_processing_mathernal_rick()
        self.npha = self.pre_processing_npha()
        self.prob_football = self.pre_processing_football()
        self.parkinsons = self.pre_processing_parkinsons()
        self.ricci = self.pre_processing_ricci()
        self.sa_heart = self.pre_processing_saHeart()
        self.german_credit = self.pre_processing_german_credit()
        self.thoracic_surgery = self.pre_processing_thoracic()
        self.wine =  self.pre_processing_wine()

    # ------------------------------------------------------------------------
    def pre_processing_wiscosin(self):
        breast_cancer_wisconsin_df = self.datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = self._prep.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )
        return breast_cancer_wisconsin_df
    
    # ------------------------------------------------------------------------
    def pre_processing_pima(self):
        pima_diabetes_df = self.datasets["pima_diabetes"].copy()
        return pima_diabetes_df
    
    # ------------------------------------------------------------------------
    def pre_processing_ricci(self):
        ricci_df = self.datasets['ricci_processed'].copy()
        ricci_df = self._prep.label_encoder(ricci_df, ["Position",
                                                       "target"])
        ricci_df = self._prep.one_hot_encode(ricci_df, ["Race"])
        return ricci_df
    
    # ------------------------------------------------------------------------
    def pre_processing_thoracic(self):
        thoracic_surgery_df = self.datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = self._prep.label_encoder(
            self._prep.one_hot_encode(thoracic_surgery_df, ["DGN"]),
            [
                "PRE7",
                "PRE8",
                "PRE9",
                "PRE10",
                "PRE11",
                "PRE17",
                "PRE19",
                "PRE25",
                "PRE30",
                "PRE32",
                "target",
            ],
        )

        thoracic_surgery_df = self._prep.ordinal_encoder(thoracic_surgery_df, ["PRE6","PRE14"])
        return thoracic_surgery_df
    
    # ------------------------------------------------------------------------
    def pre_processing_bcCoimbra(self):
        bc_coimbra = self.datasets["bc_coimbra"].copy()
        bc_coimbra = self._prep.label_encoder(bc_coimbra, ["target"])
        return bc_coimbra
    
    # ------------------------------------------------------------------------
    def pre_processing_diabetic(self):
        messidor_df = self.datasets["messidor_features"].copy()
        messidor_df["target"] = messidor_df["target"].astype("int64")
        return messidor_df 
    
    # ------------------------------------------------------------------------
    def pre_processing_indianLiver(self):
        indian_liver_df = self.datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = self._prep.label_encoder(indian_liver_df, ["Gender", "target"])
        return indian_liver_df
    
    # ------------------------------------------------------------------------
    def pre_processing_parkinsons(self):
        parkinsons_df = self.datasets["parkinsons"].copy().drop(columns="name")        
        return parkinsons_df
    
    # ------------------------------------------------------------------------
    def pre_processing_blood(self):
        blood_df = self.datasets["blood-transfusion-service-center"].copy()
        blood_df = self._prep.label_encoder(blood_df, ["target"])
        return blood_df
    
    # ------------------------------------------------------------------------
    def pre_processing_german_credit(self):
        german_credit_df = self.datasets['german'].copy()

        map_gender = {"A91":"male",
                      "A92":"female",
                      "A93":"male",
                      "A94":"male",
                      "A95":"female"}
        
        german_credit_df["personal-status-and-sex"] = german_credit_df["personal-status-and-sex"].map(map_gender)

        german_credit_df = self._prep.ordinal_encoder(german_credit_df, ["age",
                                                                         "checking-account",
                                                                         "savings-account",
                                                                         "employment-since",
                                                                         "telephone",
                                                                         "foreign-worker",
                                                                         "personal-status-and-sex"])
        german_credit_df = self._prep.label_encoder(german_credit_df, ["target"])
        
        german_credit_df = self._prep.one_hot_encode(german_credit_df, ["credit-history",
                                                                        "purpose",
                                                                        "other-debtors",
                                                                        "property",
                                                                        "other-installment",
                                                                        "housing", 
                                                                        "job"])

        return german_credit_df
    
    
    # ------------------------------------------------------------------------
    def pre_processing_acute(self):
        acute_inflammations = self.datasets["diagnosis1"].copy()
        acute_inflammations = acute_inflammations.drop(columns="target.1")
        acute_inflammations = self._prep.label_encoder(acute_inflammations, ["Nausea", 
                                                                        "Lumbar",
                                                                        "Urine",
                                                                        "Micturition",
                                                                        "Burning",
                                                                        "target"])
        return acute_inflammations
    
    # ------------------------------------------------------------------------
    def pre_processing_autism_teen(self):
        autism_adoles_df = self.datasets["Autism-Adolescent-Data"].copy()
        autism_adoles_df = autism_adoles_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"]).dropna().reset_index(drop=True)
        autism_adoles_df = self._prep.label_encoder(autism_adoles_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])
        return autism_adoles_df.astype("int64")
    
    # ------------------------------------------------------------------------
    def pre_processing_autism_child(self):
        autism_child_df = self.datasets["Autism-Child-Data"].copy()
        autism_child_df = autism_child_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"]).dropna().reset_index(drop=True)
        autism_child_df = self._prep.label_encoder(autism_child_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])
        return autism_child_df.astype("int64")
    
    # ------------------------------------------------------------------------
    def pre_processing_autism_adult(self):
        autism_df = self.datasets["Autism-Adult-Data"].copy()
        autism_df = autism_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"],
                                index = 52).dropna().reset_index(drop=True)
        autism_df = self._prep.label_encoder(autism_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])
        return autism_df.astype("int64")
    
    # ------------------------------------------------------------------------
    def pre_processing_fertility(self):
        fertility_df = self.datasets["fertility_Diagnosis"].copy()
        map_season = {-1.0:"winter",
               -0.33:"spring",
               0.33:"summer",
               -1.0:"fall"}

        map_fever = {-1.0: "less than three months ago",
                    0.0: "more than three months ago",
                    1.0: "no"}

        map_smoking = {-1.0:"never",
                    0.0:"occasional",
                    1.0: "daily"}

        fertility_df["Season"] = fertility_df["Season"].map(map_season)
        fertility_df[' high fevers'] = fertility_df[' high fevers'].map(map_fever)
        fertility_df["smoking"] = fertility_df["smoking"].map(map_smoking)

        fertility_df = self._prep.label_encoder(fertility_df, ["target"])

        fertility_df = self._prep.one_hot_encode(fertility_df, ["Season"])

        fertility_df = self._prep.ordinal_encoder(fertility_df, [' high fevers',
                                                                " alcohol consumption",
                                                                "smoking"])
        return fertility_df
    
    # ------------------------------------------------------------------------
    def pre_processing_haberman(self):
        haberman_df = self.datasets["dataset_43_haberman"].copy()
        haberman_df = self._prep.label_encoder(haberman_df, ["target"])
        return haberman_df.astype("float64")
    
    # ------------------------------------------------------------------------
    def pre_processing_mathernal_rick(self):
        maternal_health_risk_df = self.datasets["Maternal Health Risk Data Set"].copy()
        maternal_health_risk_df = self._prep.label_encoder(maternal_health_risk_df, ["target"])
        return maternal_health_risk_df
    
    # ------------------------------------------------------------------------
    def pre_processing_npha(self):
        npha_doctor_visits_df = self.datasets["NPHA-doctor-visits"].copy().drop(columns="Age")
        npha_doctor_visits_df = self._prep.label_encoder(npha_doctor_visits_df, ["target"])
        return npha_doctor_visits_df
    
    # ------------------------------------------------------------------------
    def pre_processing_saHeart(self):
        sa_heart = self.datasets["sa-heart"].copy()
        sa_heart = self._prep.label_encoder(sa_heart, ["target"])
        return sa_heart
    
    # ------------------------------------------------------------------------
    def pre_processing_HCV(self):
        hcv_egyptian_df = self.datasets["HCV-Egy-Data"].copy()
        hcv_egyptian_df = self._prep.label_encoder(hcv_egyptian_df, ["target"])
        return hcv_egyptian_df
    
    # ------------------------------------------------------------------------
    def pre_processing_iris(self):
        iris_df = self.datasets['iris'].copy()
        iris_df = self._prep.label_encoder(iris_df, ["target"])
        return iris_df
    
    # ------------------------------------------------------------------------
    def pre_processing_wine(self):
        wine_df = self.datasets['wine'].copy()
        wine_df = self._prep.label_encoder(wine_df, ["target"])
        return wine_df
    
    # ------------------------------------------------------------------------
    def pre_processing_cleveland(self):
        heart_cleveland = self.datasets['cleveland'].copy()
        heart_cleveland = self._prep.label_encoder(heart_cleveland, ["target"])
        return heart_cleveland
    
    # ------------------------------------------------------------------------
    def pre_processing_hepatitis(self):
        hepatitis = self.datasets['hepatitis'].copy()
        hepatitis = hepatitis.replace("?", np.nan).dropna()
        hepatitis = self._prep.label_encoder(hepatitis, ["target"])
        return hepatitis.astype("float64")
    
    # ------------------------------------------------------------------------
    def pre_processing_football(self):
        proba_football = self.datasets["prob_sfootball"].copy()
        proba_football = self._prep.label_encoder(proba_football, ["target"]).drop(columns=["Weekday"])
        proba_football = self._prep.ordinal_encoder(proba_football, ["Overtime"])
        proba_football = self._prep.one_hot_encode(proba_football, ["Favorite_Name",
                                                                    "Underdog_name"])

        return proba_football
    
    # ------------------------------------------------------------------------
    def pre_processing_contraceptive(self):
        contraceptive_method = self.datasets['cmc'].copy()
        contraceptive_method = self._prep.label_encoder(contraceptive_method, ["target"])
        return contraceptive_method
    
    # ------------------------------------------------------------------------
    def pre_processing_echocardiogram(self):
        echocardiogram_df = self.datasets['echocardiogram'].copy()
        echocardiogram_df = echocardiogram_df.replace("?", np.nan).dropna()
        echocardiogram_df = echocardiogram_df.drop(columns=["name","group","mult"])
        echocardiogram_df = self._prep.label_encoder(echocardiogram_df, ["target"])
        return echocardiogram_df.astype("float64")
    
    # ------------------------------------------------------------------------
    def pre_processing_phoneme(self):
        phoneme_df = self.datasets['phoneme'].copy()
        phoneme_df = self._prep.label_encoder(phoneme_df, ["target"])
        return phoneme_df
    
    # ------------------------------------------------------------------------
    def pre_processing_thyroid(self):
        thyroid_df = self.datasets["Thyroid_Diff"].copy()
        thyroid_df = self._prep.label_encoder(thyroid_df, ["target"])

        thyroid_df = self._prep.ordinal_encoder(thyroid_df, ["Physical Examination",
                                                             "Gender", 
                                                            "Smoking", 
                                                            "Hx Smoking",
                                                            "Hx Radiothreapy",
                                                            "Focality",
                                                            "Thyroid Function",
                                                            "Pathology",
                                                            "Risk",
                                                            "T",
                                                            "N",
                                                            "M",
                                                            "Stage",
                                                            "Response"])

        thyroid_df = self._prep.one_hot_encode(thyroid_df, ["Adenopathy"])
        return thyroid_df
    
    # ------------------------------------------------------------------------
    def pre_processing_bank(self):
        bank_marketing = self.datasets['bank_marketing'].copy()

        bank_marketing = self._prep.one_hot_encode(bank_marketing, ["V4",
                                                                    "V3",
                                                                    "V2",
                                                                    "V9",
                                                                    "V11",
                                                                    "V16"])
        bank_marketing = self._prep.ordinal_encoder(bank_marketing, ["V5",
                                                                     "V7",
                                                                     "V8"])
        bank_marketing = self._prep.label_encoder(bank_marketing, ["target"])
        return bank_marketing
    
    # ------------------------------------------------------------------------
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [self.acute,
                                         self.autism_teen,
                                         self.autism_adult,
                                         self.autism_child,
                                         self.bc_coimbra,
                                         self.blood_transfusion,
                                         self.contraceptive,
                                         self.diabetic,
                                         self.echocardiogram,
                                         self.fertility,
                                         self.german_credit,
                                         self.haberman,
                                         self.hcv,
                                         self.cleveland,
                                         self.hepatitis,
                                         self.iris,
                                         self.liver,
                                         self.mathernal_risk,
                                         self.npha,
                                        #  self.parkinsons,
                                         self.phoneme,
                                         self.pima,
                                         self.prob_football,
                                         self.ricci,
                                         self.sa_heart,
                                         self.thoracic_surgery,
                                         self.thyroid,
                                        #  self.wine,
                                        #  self.wiscosin                                         
                                         ]
            
        tabela_resultados["nome_datasets"] = ["acute-inflammations",
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
                                            #   "parkinsons",
                                              "phoneme",
                                              "pima-diabetes",
                                              "proba_football",
                                              "ricci",
                                              "sa-heart",
                                              "thoracic-surgery",
                                              "thyroid-recurrence",
                                            #   "wine",
                                            #   "wiscosin"
                                              ]
            
        tabela_resultados["missing_rate"] = [5,20,40]

        return tabela_resultados
    
    # ------------------------------------------------------------------------
    def cria_tabela_cybersecurity(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [self.datasets["nsl_kdd_proceed"]
                                         ]
            
        tabela_resultados["nome_datasets"] = ["nsl_kdd"]
            
        tabela_resultados["missing_rate"] = [5,20,40]

        return tabela_resultados
    
    # ------------------------------------------------------------------------
    def find_duplicates(x_train, y_train, columns):
        """
        Returns an array of booleans that is true if that element was previously in the array

        :param x_train: training data
        :type x_train: `np.ndarray`
        :return: duplicates array
        :rtype: `np.ndarray`
        """
        x_train = pd.DataFrame(x_train, columns = columns)
        x_train["target"] = y_train
        
        x_train.drop_duplicates(inplace=True)
        
        return x_train
    
    # ------------------------------------------------------------------------
    def get_data(X_raw:pd.DataFrame,
                 y_raw:np.ndarray
                 ):  
        
        X_raw_columns = X_raw.columns
        dups = AdversarialML.find_duplicates(X_raw, y_raw, X_raw_columns)
        
        X_ = dups.drop(columns="target")
        y_ = dups["target"].values
        
        X_raw = X_.astype("float64").values
        y = np.copy(y_)
        num_classes = 40
                
        labels = np.zeros((y.shape[0], num_classes))
        
        labels[np.arange(y.shape[0]), y] = 1
        
        y = labels
        n_sample = len(X_raw)
    
        order = np.random.permutation(n_sample)
        
        X = X_raw[order]
        y = y[order]
        
        return X, y
    
    def train_and_save(X_train:pd.DataFrame,
                        y_train:np.ndarray,
                        folder:int) -> SVC:
        """
        Train an SVM model using the given training data and save it to a specified folder.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature set.
        y_train : np.ndarray
            The training target labels.
        folder : int
            Identifier for the folder where the model will be saved.

        Returns
        -------
        SVC
            The trained SVM model.
        """
                
        svm = SVC(probability=True, random_state=42)

        # 3. Definir o espaço de busca de hiperparâmetros
        param_distributions = {
            "C": np.arange(0.1, 10),         # C: Regularização
            "kernel": ["linear", "rbf"],   # Kernel: linear ou radial
            "gamma": np.arange(0.01, 1),     # Gamma (apenas para kernels não lineares)
        }

        # 4. Configurar o RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=svm,
            param_distributions=param_distributions,
            n_iter=5,                # Número de combinações aleatórias testadas
            scoring="f1_weighted",       # Métrica de avaliação
            verbose=2,                # Verbosidade para acompanhar o progresso
            random_state=42,          # Reprodutibilidade
            n_jobs=-1                 # Paralelização para acelerar o treinamento
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        # Save the model
        joblib.dump(best_model,f"./models/svm_folder{folder}.pkl")

        return best_model
    
    def get_model(folder:int)-> SVC:
        """
        Load a saved SVM model from the specified folder.

        Parameters
        ----------
        folder : int
            Identifier for the folder where the model is stored.

        Returns
        -------
        SVC
            The loaded SVM model.
        """
        path = f"./models/svm_folder{folder}.pkl"
        model = joblib.load(path)
        return model
    
    def return_art_classifier(X_train:pd.DataFrame,
                              y_train:np.ndarray,
                              X_test:pd.DataFrame,
                              folder:int = None):
        
        """
        Create and return an ART classifier based on a trained SVM model and test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            The test feature set, which will be converted to float64 type.
        folder : int
            Identifier for the folder where the SVM model is stored.

        Returns
        -------
        tuple
            A tuple containing:
            - SklearnClassifier: An ART classifier wrapping the SVM model.
            - np.ndarray: The test data converted to a NumPy array.
        """
        
        X_test_float = X_test.astype("float64")
        x_selected_adv = X_test_float.values
        min_value = min(X_test_float.min())
        max_value = max(X_test_float.max())

        # model = AdversarialML.get_model(folder)
        # Create and fit the Scikit-learn model
        model = SVC(C=1.0, kernel="rbf", probability=True)
        model.fit(X=X_train, y=y_train)

        # Create ART classifier for scikit-learn SVC
        art_classifier = SklearnClassifier(model=model, clip_values=(min_value, max_value))

        return art_classifier, x_selected_adv

    
    # ------------------------------------------------------------------------
    @staticmethod
    def FGSM(X_train:pd.DataFrame,
            y_train:np.ndarray, 
            X_test:pd.DataFrame):
        
        """
        Generate adversarial attack using Fast Sign Grandient Method (FGSM)

        Args:

        Returns:

        """
        # art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_test, folder)

        # attack = FastGradientMethod(estimator=art_classifier, 
        #                             eps=0.3,
        #                             norm=np.inf,
        #                             batch_size=128
        #                             )
        # x_adv = attack.generate(x=x_adv_f)

        X_test_float = X_test.astype("float64")
        x_selected_adv = X_test_float.values
        min_value = min(X_test_float.min())
        max_value = max(X_test_float.max())

        # Create and fit the Scikit-learn model
        model = SVC(C=1.0, kernel="rbf", probability=True)
        model.fit(X=X_train, y=y_train)

        # Create ART classifier for scikit-learn SVC
        art_classifier = SklearnClassifier(model=model, clip_values=(min_value, max_value))

        attack = FastGradientMethod(estimator=art_classifier, 
                                    eps=0.3,
                                    norm=np.inf)
        x_adv = attack.generate(x=x_selected_adv)

        return pd.DataFrame(x_adv, columns=X_test.columns)

    # ------------------------------------------------------------------------
    @staticmethod
    def attack_poison(X_train:pd.DataFrame,
                      y_train:np.ndarray,
                      X_test:pd.DataFrame,
                      y_test:np.ndarray):
        """
        Generate adversarial examples on a given dataset.

        Args:
            X (pd.DataFrame): The original dataset.
            y (np.array): The target values.
            noise_level (int): The noise ratio for generating adversarial examples, given as a percentage (0-100).

        Returns:
            pd.DataFrame: A DataFrame containing the dataset with adversarial examples added.
        """

        art_classifier, _ = AdversarialML.return_art_classifier(X_train, y_train, X_test)
        
        X_train_format, y_train_format = AdversarialML.get_data(X_train, y_train)
        X_test_format, y_test_format = AdversarialML.get_data(X_test, y_test)
        
        attack_idx = np.random.choice(len(X_train_format))
        
        init_attack = np.copy(X_train_format[attack_idx])
        
        y_attack = np.ones(y_train_format.shape[1]) - np.copy(y_train_format[attack_idx])

        attack = PoisoningAttackSVM(classifier=art_classifier, 
                                    step=0.1, 
                                    eps = 1.0, 
                                    x_train= X_train_format, 
                                    y_train= y_train_format, 
                                    x_val = X_test_format,
                                    y_val =  y_test_format,
                                    max_iter=10)
                                    
        x_adv, y_adv  = attack.poison(np.array([init_attack]), np.array([y_attack]))
        x_result = pd.concat([X_train, pd.DataFrame(x_adv, columns=X_train.columns)]).reset_index(drop=True)
        
        if np.sum(y_adv[0]) > 1:
            y_adv_result = np.argmin(y_adv[0])
        elif np.sum(y_adv[0]) == 1:
            y_adv_result = np.argmax(y_adv[0])
        
        return pd.DataFrame(x_result, columns=X_train.columns), y_adv_result

    # ------------------------------------------------------------------------
    @staticmethod
    def PGD(X_train:pd.DataFrame,
            y_train:np.ndarray,
            X_test:pd.DataFrame):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_train, y_train, X_test)
        attack = ProjectedGradientDescent(estimator=art_classifier,
                                          norm=2,
                                          max_iter=10,
                                          eps=0.15,
                                          batch_size=32)
        
        x_adv = attack.generate(x=x_adv_f)

        return pd.DataFrame(x_adv, columns=X_test.columns)
    
    # ------------------------------------------------------------------------
    @staticmethod
    def Carlini(X_train:pd.DataFrame,
            y_train:np.ndarray,
            X_test:pd.DataFrame):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_train, y_train, X_test)

        attack = CarliniL2Method(classifier=art_classifier,
                                 batch_size=32,
                                 max_iter=10)
        
        x_adv = attack.generate(x_adv_f)

        return pd.DataFrame(x_adv, columns=X_test.columns)