import pickle
import pandas as pd
import utilities as utl
import data_wrangling as dw
import machinelearn2 as mlearn
from sklearn import svm
# =============================================================================
# Add teams + round num
# =============================================================================
teams = ['Richmond', 'Sydney','Essendon',  'Port Adelaide', 'Geelong', 'West Coast', 'North Melbourne','Hawthorn','Gold Coast']
opponents = ['Collingwood', 'Adelaide','St Kilda','Carlton','Melbourne','GWS','Brisbane','Western Bulldogs',  'Fremantle']
round_n = 2

year = 2019
games = range(1,len(teams)+1)

eloCreate(afl_DF, colTeam, colOpponent, colHomeGame, colWin, eloInitialScore, HGA, K, m, colEloOut, colSeason, regressParam ):

# =============================================================================
# Create a new round to add all the metrics to
# =============================================================================
afl_NEW = utl.createScoringData(teams, opponents, games, round_n, year)
# =============================================================================
# Run the averaging to populate the new round
# =============================================================================
afl_NEW = dw.dataWrangling(new_data = afl_NEW,  scoring_data = True)
# =============================================================================
# Apply data transformations 
# =============================================================================
X_cols, X_scaler, feat_selector = pickle.load(open("objectsScore.pkl", 'rb')) 

# =============================================================================
# Scoring Model 1
# =============================================================================
SVC_clf = pickle.load(open("SVC_clf.pkl", 'rb')) 
afl_SCORED_SVM = mlearn.dataScored(afl_NEW, X_cols, X_scaler, feat_selector, SVC_clf)
# =============================================================================
# Scoring Model 2
# =============================================================================
RF_clf = pickle.load(open("RF_clf.pkl", 'rb')) 
afl_SCORED_RF = mlearn.dataScored(afl_NEW, X_cols, X_scaler, feat_selector, RF_clf ) 
# =============================================================================
# Getting an overall average model score
# =============================================================================
mlearn.avgModel(afl_SCORED_SVM, afl_SCORED_RF)


SVC_clf