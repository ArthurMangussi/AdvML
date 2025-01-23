# Studying the Robustness of Data Imputation Methodologies Against Adversarial Attacks

This repository contains the codebase for the paper: *Studying the Robustness of Data Imputation Methodologies Against Adversarial Attacks*

## Paper Details
- Authors: Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Ana Carolina Lorena, Miriam Seoane Santos, and Pedro Henriques Abreu
- Abtract: Cybersecurity attacks, such as poisoning and evasion, can intentionally introduce
false or misleading information in different forms into data, potentially
leading to catastrophic consequences for critical infrastructures, like water
supply or energy power plants. While numerous studies have investigated
the impact of these attacks on model-based prediction approaches, they often
overlook the impurities present in the data used to train these models.
One of those forms is missing data, the absence of values in one or more
features. This issue is typically addressed by imputing missing values with
plausible estimates, which directly impacts the performance of the classifier.
The goal of this work is to promote a Data-centric AI approach by investigating
how different types of cybersecurity attacks impact the imputation
process. To this end, we conducted experiments using two popular evasion
and poisoning attacks strategies across 29 real-world datasets. For the adversarial attack strategies, we employed the Fast Gradient Sign Method and Poison Attack against Support Vector Machine algorithm. Also, four state-of-
the-art imputation strategies were tested under Missing Not At Random
and Missing At Random mechanisms using three missing rates (5%, 20%,
40%). We assessed imputation quality using Mean Absolute Error, Kolmogorov–
Smirnov test, and Chi-Squared test. Our findings demonstrate
that adversarial attacks significantly impact the imputation process. In
terms of imputation quality error, the SoftImpute outperforms other imputation
strategies. Regarding data distribution error, results from the Kolmogorov–
Smirnov test indicate that all imputation strategies generate numerical
features that differ from the baseline. However, the Chi-Squared test
shows that categorical features after imputation are not statistically different
from the baseline.
- Keywords: Data-centric AI, Cybersecurity attacks, Missing Data Imputation, Adversarial Machine Learning
- Year: 2025
- Contact: mangussiarthur@gmail.com

## Installation
```bash
git clone https://github.com/ArthurMangussi/AdvML.git
cd AdvML
pip install -r requirements.txt
```

## Reproducibility
Follow the steps below to reproduce the experiments:
1. **Run the baseline experiments**
Execute the missing data imputation methods without applying any adversarial attacks:
```bash
python baseline_mar.py
python baseline_mnar.py
python baseline_mcar.py
```
2. **Run the adversarial attack experiments**
For each combination of attack type (poisoning or evasion) and missing data mechanism (MAR or MNAR), run the following command:
```bash
python adversarial_ADV_MD.py
```
Replace ```ADV``` with ```poison``` or ```evasion```, and ```MD``` with ```mar``` or ```mnar``` accordingly.

3. **Combine the final results table**
Combine the results into a unified table using the script:
```bash
python gera_tabela_final.py
```
4. **Unify test folds**
The test folds are saved during the main logic execution. Use the following script to merge them:
```bash
python unifica_folds.py
```
5. **Evaluate data distributions**
Perform the Kolmogorov-Smirnov and Chi-square tests to assess the data distributions:
```bash
python ktest.py
```

## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) under grants 2021/06870-3, 2022/10553-6, and 2023/13688-2. Moreover, this research was supported in part by Portuguese Recovery and Resilience Plan (PRR) through project C645008882-00000055 Center for Responsable AI.
