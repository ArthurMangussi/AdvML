# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from utilsMsc.MyResults import AnalysisResults


tipo = 'Multivariado'
folder = "@MAR-random"
mecanismo = folder+'_'+tipo

for i in ["MAR-random"]:
    r = AnalysisResults()
    path_dumb = f'mean_{i}'
    path_knn = f'customKNN_{i}'
    path_mice = f'mice_{i}'
    path_pmivae = f'pmivae_{i}'
    path_saei = f'saei_{i}'
    path_soft_impute = f'softImpute_{i}'
    path_gain = f'gain_{i}'
    path_missForest = f'missForest_{i}'
    path_baye = f'bayesian_{i}'

    r.gera_tabela_unificada(
        tipo,
        mecanismo,
        path_dumb,
        path_knn,
        path_mice,
        path_pmivae,
        path_soft_impute,
        path_gain,
        #path_baye,
        #path_missForest
    )
