from utilsMsc.MyPreprocessing import PreprocessingDatasets
from utilsMsc.MeLogSingle import MeLogger

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

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

        tabela_resultados["datasets"] = [self.datasets["nsl_proceed"],
                                         self.datasets["preprocessed_DNN"]]
            
        tabela_resultados["nome_datasets"] = ["nsl_kdd",
                                              "edge"]
            
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
    
    def return_art_classifier(X_train:pd.DataFrame,
                        y_train:np.ndarray, 
                        X_test:pd.DataFrame,
                        ):
        
        X_test_float = X_test.astype("float64")
        x_selected_adv = X_test_float.values
        min_value = min(X_test_float.min())
        max_value = max(X_test_float.max())

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
                        X_test:pd.DataFrame,
                        y_test:np.ndarray):
        
        """
        Generate adversarial attack using Fast Sign Grandient Method (FGSM)

        Args:

        Returns:

        """
        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_train, y_train, X_test)

        attack = FastGradientMethod(estimator=art_classifier, 
                                    eps=0.3,
                                    norm=np.inf)
        x_adv = attack.generate(x=x_adv_f)

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
    def PGD(X_train:pd.DataFrame,
                      y_train:np.ndarray,
                      X_test:pd.DataFrame,
                      y_test:np.ndarray):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_train, y_train, X_test)
        attack = ProjectedGradientDescent(estimator=art_classifier,
                                          norm=2,
                                          max_iter=10)
        
        x_adv = attack.generate(x=x_adv_f)

        return x_adv
    
    # ------------------------------------------------------------------------
    @staticmethod
    def Carlini(X_train:pd.DataFrame,
                      y_train:np.ndarray,
                      X_test:pd.DataFrame,
                      y_test:np.ndarray):

        art_classifier, x_adv_f = AdversarialML.return_art_classifier(X_train, y_train, X_test)
        attack = CarliniL2Method(classifier=art_classifier,
                                 batch_size=32,
                                 max_iter=10)
        
        x_adv = attack.generate(x_adv_f, y_test)

        return x_adv