import pickle
import dataWrangling as dw
import utilities.utilitiesML as mlearn
import pandas as pd
# =============================================================================
# Set up the inputs for the modelling
# =============================================================================
year_weights = {2008: 1,
 2009: 1,
 2010: 1,
 2011: 1,
 2012: 1,
 2013: 1,
 2014: 1,
 2015: 1,
 2016: 1,
 2017: 1,
 2018: 1}
M_cols = ['fw_game_id', 'year', 'round_index','opponent', 'team']
test_ids_file = 'test_fw_ids.pkl'
target = 'model_target'
# =============================================================================
# Import data, define predictor columns and split test and train
# =============================================================================
afl_ML, afl_DF = dw.dataWrangling(new_data = '', outputType = '')
dfFantasyAverage = pd.read_pickle('data/dfFantasyAverage.pkl')

###Add in fantasy###
afl_ML['fw_game_id'] = afl_ML['fw_game_id'].astype('int')
dfFantasyAverage_ML = afl_ML[['fw_game_id','team','opponent']]

dfFantasyAverage_team = dfFantasyAverage
dfFantasyAverage_team.columns =['fw_game_id', 'team', 'supercoachavg_team', 'aflfantasyavg_team']
dfFantasyAverage_ML = pd.merge(dfFantasyAverage_ML ,
      dfFantasyAverage_team,
      left_on = ['team','fw_game_id'],
      right_on = ['team','fw_game_id'],
      how = 'left')

dfFantasyAverage_opponent = dfFantasyAverage
dfFantasyAverage_opponent.columns =['fw_game_id', 'opponent', 'supercoachavg_opponent', 'aflfantasyavg_opponent']
dfFantasyAverage_ML = pd.merge(dfFantasyAverage_ML,
      dfFantasyAverage_opponent,
      left_on = ['opponent','fw_game_id'],
      right_on = ['opponent','fw_game_id'],
      how = 'left')

dfFantasyAverage_ML['fantasyavg_team'] = dfFantasyAverage_ML['aflfantasyavg_team'] + dfFantasyAverage_ML['supercoachavg_team']
dfFantasyAverage_ML['fantasyavg_opponent'] = dfFantasyAverage_ML['aflfantasyavg_opponent'] + dfFantasyAverage_ML['supercoachavg_opponent']

dfFantasyAverage_ML['SupercoachAvg_D'] = (dfFantasyAverage_ML['supercoachavg_team'] / (dfFantasyAverage_ML['supercoachavg_team'] + dfFantasyAverage_ML['supercoachavg_opponent']))
dfFantasyAverage_ML['AflFantasyAvg_D'] = (dfFantasyAverage_ML['aflfantasyavg_team'] / (dfFantasyAverage_ML['aflfantasyavg_team'] + dfFantasyAverage_ML['aflfantasyavg_opponent']))
dfFantasyAverage_ML['FantasyAvg_D'] = (dfFantasyAverage_ML['fantasyavg_team'] / (dfFantasyAverage_ML['fantasyavg_team'] + dfFantasyAverage_ML['fantasyavg_opponent']))

dfFantasyAverage_ML = dfFantasyAverage_ML[['fw_game_id','team', 'FantasyAvg_D']]
afl_ML = pd.merge(afl_ML,
         dfFantasyAverage_ML,
         on = ['fw_game_id','team'],
         how = 'left')

###Define ML Data###
X_cols = mlearn.defineMLData(afl_ML, target, M_cols)
test_ids, train, test = mlearn.splitData(afl_ML, test_ids_file)
afl_ML.columns
# =============================================================================
# Data transformation
# =============================================================================
X_scaler = mlearn.scaleMLData(train, X_cols)

X_test, X_test_scaled, X_train, X_train_scaled, Y_test, Y_train, Y_trainval, M_train, M_test, target, sample_weights = mlearn.prepareMLData(train, test, target, X_cols, M_cols, year_weights, X_scaler)
# =============================================================================
# Feature selection
# =============================================================================
X_test_select, X_train_select, X_selectColnames, feat_selector = mlearn.selBoruta(X_test, X_test_scaled, X_train, X_train_scaled, Y_trainval)

objectsTraining = (X_train_select, Y_train, sample_weights)
pickle.dump(objectsTraining , open("objectsTraining.pkl", 'wb'))

objectsTest = (X_test_select, Y_test)
pickle.dump(objectsTest , open("objectsTest.pkl", 'wb'))

objectsScore = (X_cols, X_scaler, feat_selector)
pickle.dump(objectsScore , open("objectsScore.pkl", 'wb'))

pickle.dump(feat_selector, open("featureSelect.pkl", 'wb'))

