from numpy.random.mtrand import uniform
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
    
    # ------------------------------------------------------------------------
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [self.datasets["nsl_kdd_proceed"]
                                         ]
            
        tabela_resultados["nome_datasets"] = ["nsl_kdd"]
            
        tabela_resultados["missing_rate"] = [5,20,40]

        return tabela_resultados
    
    # ------------------------------------------------------------------------
    def find_duplicates(x_train):
        """
        Returns an array of booleans that is true if that element was previously in the array

        :param x_train: training data
        :type x_train: `np.ndarray`
        :return: duplicates array
        :rtype: `np.ndarray`
        """
        dup = np.zeros(x_train.shape[0])
        for idx, x in enumerate(x_train):
            dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
        return dup
    
    # ------------------------------------------------------------------------
    def get_data(X_raw:pd.DataFrame,
                 y_raw:np.ndarray,
                 ):  
        
        X_raw = X_raw.astype("float64").values
        y = np.copy(y_raw)
                
        if len(np.unique(y)) == 2:
            labels = np.zeros((y.shape[0], 2))  
            labels[y == 0] = np.array([1, 0])
            labels[y == 1] = np.array([0, 1])
        elif len(np.unique(y)) == 3:
            labels = np.zeros((y.shape[0], 3)) 
            labels[y == 0] = np.array([1, 0, 0])
            labels[y == 1] = np.array([0, 1, 0])
            labels[y == 2] = np.array([0, 0, 1])
        elif len(np.unique(y)) == 4:
            labels = np.zeros((y.shape[0], 4)) 
            labels[y == 0] = np.array([1, 0, 0, 0])
            labels[y == 1] = np.array([0, 1, 0, 0])
            labels[y == 2] = np.array([0, 0, 1, 0])
            labels[y == 3] = np.array([0, 0, 0, 1])
                        
        y = labels
        n_sample = len(X_raw)
    
        order = np.random.permutation(n_sample)
        
        X = X_raw[order]
        y = y[order]

        dups = AdversarialML.find_duplicates(X_raw)
        X_format = X[dups == False]
        y_format = y[dups == False]
        
        return X_format, y_format
    
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
            "C": uniform(0.1, 10),         # C: Regularização
            "kernel": ["linear", "rbf"],   # Kernel: linear ou radial
            "gamma": uniform(0.01, 1),     # Gamma (apenas para kernels não lineares)
        }

        # 4. Configurar o RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=svm,
            param_distributions=param_distributions,
            n_iter=50,                # Número de combinações aleatórias testadas
            scoring="f1_weighted",       # Métrica de avaliação
            cv=5,                     # Cross-validation (5 folds)
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
    
    def return_art_classifier(X_test:pd.DataFrame,
                              folder:int):
        
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

        model = AdversarialML.get_model(folder)

        # Create ART classifier for scikit-learn SVC
        art_classifier = SklearnClassifier(model=model, clip_values=(min_value, max_value))

        return art_classifier, x_selected_adv

    
    # ------------------------------------------------------------------------
    @staticmethod
    def FGSM(X_test:pd.DataFrame,
             folder:int):
        
        """
        Generate adversarial attack using Fast Sign Grandient Method (FGSM)

        Args:

        Returns:

        """
        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_test, folder)

        attack = FastGradientMethod(estimator=art_classifier, 
                                    eps=0.3,
                                    norm=np.inf,
                                    batch_size=128
                                    )
        x_adv = attack.generate(x=x_adv_f)

        return pd.DataFrame(x_adv, columns=X_test.columns)

    # ------------------------------------------------------------------------
    @staticmethod
    def attack_poison(X_train:pd.DataFrame,
                      y_train:np.ndarray,
                      X_test:pd.DataFrame,
                      y_test:np.ndarray,
                      folder:int):
        """
        Generate adversarial examples on a given dataset.

        Args:
            X (pd.DataFrame): The original dataset.
            y (np.array): The target values.
            noise_level (int): The noise ratio for generating adversarial examples, given as a percentage (0-100).

        Returns:
            pd.DataFrame: A DataFrame containing the dataset with adversarial examples added.
        """
        
        art_classifier, _ = AdversarialML.return_art_classifier(X_test,folder)
        
        X_train_format, y_train_format = AdversarialML.get_data(X_train, y_train)
        X_test_format, y_test_format = AdversarialML.get_data(X_test, y_test)
        attack_idx = np.random.choice(len(X_train_format))

        init_attack = np.copy(X_train_format[attack_idx])
        if y_train_format.shape[1] == 2:
            y_attack = np.array([1, 1]) - np.copy(y_train_format[attack_idx])
        elif y_train_format.shape[1] == 3:
            y_attack = np.array([1, 1, 1]) - np.copy(y_train_format[attack_idx])
        elif y_train_format.shape[1] == 4:
            y_attack = np.array([1, 1, 1, 1]) - np.copy(y_train_format[attack_idx])
        else:
            raise ValueError("Deu erro no shape do y_train_format")

        attack = PoisoningAttackSVM(classifier=art_classifier, 
                                    step=0.001, 
                                    eps = 1.0, 
                                    x_train= X_train_format, 
                                    y_train= y_train_format, 
                                    x_val= X_test_format, 
                                    y_val= y_test_format, 
                                    max_iter=10)
        x_adv, y_adv = attack.poison(np.array([init_attack]), y=np.array([y_attack]))
        x_result = pd.concat([X_train, pd.DataFrame(x_adv, columns=X_train.columns)])

        return x_result, y_adv[0][1]

    # ------------------------------------------------------------------------
    @staticmethod
    def PGD(X_test:pd.DataFrame,
            folder:int):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_test,folder)
        attack = ProjectedGradientDescent(estimator=art_classifier,
                                          norm=2,
                                          max_iter=10)
        
        x_adv = attack.generate(x=x_adv_f)

        return x_adv
    
    # ------------------------------------------------------------------------
    @staticmethod
    def Carlini(X_test:pd.DataFrame,
                y_test:np.ndarray,
                folder:int):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_test,folder)
        attack = CarliniL2Method(classifier=art_classifier,
                                 batch_size=32,
                                 max_iter=10)
        
        x_adv = attack.generate(x_adv_f, y_test)

        return x_adv