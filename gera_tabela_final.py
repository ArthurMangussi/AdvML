# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from utilsMsc.MyResults import AnalysisResults


type_attack = 'baseline'

for MD_mechanism in ["MAR-correlated", "MNAR-determisticTrue"]:
    r = AnalysisResults()
    path_knn = f'knn_{MD_mechanism}'
    path_mice = f'mice_{MD_mechanism}'
    path_soft_impute = f'softImpute_{MD_mechanism}'
    path_gain = f'gain_{MD_mechanism}'

    r.gera_tabela_unificada(
        type_attack,
        MD_mechanism,
        path_knn,
        path_mice,
        path_soft_impute,
        path_gain,
    )
