#%%
#Setup
first_year = 2012 #earliest year to include in data wrangling. Zero means, all of them.
#note: after running averages this will mean the earliest year in the model will be later than this
exclude_finals = 1 #1 = exclude, 2 = include
games_to_keep = 'mix' #'mix' = mix of home and away, 'home' 'away' = home and away respectively. Else
data_file = 'afl_DF.pkl'

target = 'Win'
M_cols = ['fw_game_id', 'year', 'round_index']
test_ids_file = 'test_fw_ids.pkl'
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

score = 'neg_log_loss'

import data_wrangling as dw
import machinelearn2 as mlearn
#%%
afl_DF, afl_ML, afl_NEW = dw.dataWrangle(first_year, exclude_finals, games_to_keep, data_file)
#%%
test_ids, train, test = mlearn.splitData(afl_ML, test_ids_file)
#%%
X_cols, X_scaler = mlearn.defineMLData(train, target, M_cols)
#%%
X_test, X_test_scaled, X_train, X_train_scaled, Y_test, Y_train, Y_trainval, M_train, M_test, target, sample_weights = mlearn.prepareMLData(train, test, target, X_cols, M_cols, year_weights, X_scaler)
#%%
X_test_boruta, X_train_boruta, feat_selector, X_borutaColnames = mlearn.selBoruta(X_test, X_test_scaled, X_train, X_train_scaled, Y_trainval)
#,
#                                             n_jobs=-1, class_weight='balanced', max_depth=5, #rf classifier
#                                             n_estimators='auto', verbose=2, random_state=1, #boruta inputs
#                                             rseed = 80085)

#%%
svm_param_set = [{'C':[0.00001, 0.0001, 0.001, 0.1, 0.5], 'kernel': ['linear']},
             {'C': [0.1, 0.5, 1, 5, 10], 'gamma': [0.01, 0.001,0.0005, 0.0001,0.00005,0.00001], 'kernel': ['rbf']}]
SVC_clf = mlearn.trainSVC(X_train_boruta, Y_train, svm_param_set , score, sample_weights)
SVC_clf.grid_scores_
#%%
SVC_clf.best_estimator_
#%%
monash_scoreSVM = mlearn.scoreMonash(SVC_clf , X_test_boruta, Y_test)
#%%
print(sum(monash_scoreSVM .monash_score > 0)/len(monash_scoreSVM ),
sum(monash_scoreSVM .monash_score))
#%%
afl_SCORED_SVM = mlearn.dataScored(afl_NEW, X_cols, X_scaler, feat_selector, X_borutaColnames, SVC_clf)
afl_SCORED_SVM 
#%%
rf_param_set = {
       #'bootstrap': [True, False],
       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        }
RF_clf = mlearn.trainRandRF(X_train_boruta, Y_train, rf_param_set , score, sample_weights, 100)
#%%
RF_clf .grid_scores_
#%%
RF_clf.best_estimator_
#%%
monash_scoreRF = mlearn.scoreMonash(RF_clf, X_test_boruta, Y_test)

#%%
print(sum(monash_scoreRF.monash_score > 0)/len(monash_scoreRF),
sum(monash_scoreRF.monash_score))
#%%
afl_SCORED_RF = mlearn.dataScored(afl_NEW, X_cols, X_scaler, feat_selector, X_borutaColnames, RF_clf)
afl_SCORED_RF 
#%%
import numpy as np
monash_scoreRF['SVM_prob'] = monash_scoreSVM[1]
monash_scoreRF['avg_prob'] = (monash_scoreRF['SVM_prob'] + monash_scoreRF[1]) / 2
monash_scoreRF['avg_monash_score'] = ((1 + np.log2(monash_scoreRF['avg_prob'])) * monash_scoreRF['Win']) + (1 + np.log2(1 - monash_scoreRF['avg_prob'])) * (1 - monash_scoreRF['Win'])
sum(monash_scoreRF['avg_monash_score'])
#%%
mlearn.scoreOddsBenchmark(test_ids_file, afl_DF)
