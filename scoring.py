
import pickle
import datetime as dt
import utilities.utilities as utl
import dataWrangling as dw
import utilities.utilitiesML as mlearn
import utilities.utilitiesElo as eloUtl
# =============================================================================
# Add teams + round num
# =============================================================================
teams = ['Brisbane', 'North Melbourne','West Coast','GWS','Melbourne','Richmond',  'Western Bulldogs', 'Adelaide', 'Hawthorn']
opponents = ['Collingwood', 'Essendon', 'Port Adelaide','Fremantle', 'St Kilda' ,   'Sydney', 'Carlton',  'Gold Coast', 'Geelong']
round_n = 2

year = 2019
games = range(1,len(teams)+1)
# =============================================================================
# Create a new round to add all the metrics to
# =============================================================================
afl_NEWentry = utl.createScoringData(teams, opponents, games, round_n, year)
# =============================================================================
# Run the averaging to populate the new round
# =============================================================================
afl_NEW = dw.dataWrangling(new_data = afl_NEWentry,  outputType = 'score')
afl_NEW_ELO = dw.dataWrangling(new_data = afl_NEWentry,  outputType = 'scoreElo')
# =============================================================================
# Create a eloScores
# =============================================================================
afl_SCORED_ELO = eloUtl.eloCreate(afl_NEW_ELO , eloInitialScore = 1500, HGA = 40 , K = 50, m = 400, regressParam = 0.5, p = 0.04, KMargin = 0.3, colMargin = 'Margin', colTeam = 'team', colOpponent = 'opponent', colHomeGame = 'home_game', colWin = 'model_target', colEloOut = 'eloProb', colSeason = 'year')

afl_SCORED_ELO = afl_SCORED_ELO.loc[afl_SCORED_ELO['date'] == dt.datetime(2099, 12, 31),]
afl_SCORED_ELO = afl_SCORED_ELO[['team', 'opponent', 'home_game', 'eloProb']]

afl_index = afl_SCORED_ELO.home_game == 1
afl_SCORED_ELO[['home_team']] = afl_SCORED_ELO[['team']]
afl_SCORED_ELO[['Win']] = afl_SCORED_ELO[['eloProb']]
afl_SCORED_ELO.loc[~afl_index,'home_team'] = afl_SCORED_ELO.loc[~afl_index, 'opponent']
afl_SCORED_ELO.loc[~afl_index,'Win'] = 1 - afl_SCORED_ELO.loc[~afl_index, 'eloProb']
afl_SCORED_ELO ['team'] = afl_SCORED_ELO['home_team']
afl_SCORED_ELO = afl_SCORED_ELO[['team','Win']]
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
mlAvgScore = mlearn.avgModel(afl_SCORED_SVM, afl_SCORED_RF) # Average of all the ML models
cmbAvgScore = mlearn.avgModel(mlAvgScore, afl_SCORED_ELO) # Average of ML and ELO models

pickle.dump(cmbAvgScore, open("cmbAvgScore.pkl", 'wb'))