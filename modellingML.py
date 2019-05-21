import utilities.utilitiesML as mlearn
import pickle
# =============================================================================
# Scoring metric
# =============================================================================
score = 'neg_log_loss' 
# =============================================================================
# Load in training data
# =============================================================================
X_train_select, Y_train, sample_weights = pickle.load(open("objectsTraining.pkl", 'rb')) 
# =============================================================================
# 
# =============================================================================
svm_param_set = [{'C':[0.00001, 0.0001, 0.001, 0.1, 0.5, 1, 2, 5], 'kernel': ['linear']},
             {'C': [0.0001, 0.001, 0.1, 0.5, 1, 5, 10], 'gamma': [0.1, 0.01, 0.001,0.0001,0.00001], 'kernel': ['rbf']}]
SVC_clf = mlearn.trainSVC(X_train_select, Y_train, svm_param_set , score, sample_weights)

pickle.dump(SVC_clf, open("SVC_clf.pkl", 'wb'))
SVC_clf = pickle.load(open("SVC_clf.pkl", 'rb')) 

# =============================================================================
# 
# =============================================================================
rf_param_set = {
       #'bootstrap': [True, False],
       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        }
RF_clf = mlearn.trainRandRF(X_train_select, Y_train, rf_param_set , score, sample_weights, 200)

pickle.dump(RF_clf, open("RF_clf.pkl", 'wb'))
RF_clf = pickle.load(open("RF_clf.pkl", 'rb')) 
RF_clf.best_estimator_.feature_importances_
list(X_train_select.columns[])
# =============================================================================
# 
# =============================================================================
X_test_select, Y_test= pickle.load(open("objectsTest.pkl", 'rb'))

monash_scoreSVM = mlearn.scoreMonash(SVC_clf , X_test_select, Y_test)
monash_scoreRF = mlearn.scoreMonash(RF_clf , X_test_select, Y_test)

scoreProbs(SVC_clf , X_test_select, Y_test)
print('SVM', sum(monash_scoreSVM .monash_score > 0)/len(monash_scoreSVM ),
sum(monash_scoreSVM .monash_score))

import numpy as np
print('RF', sum(monash_scoreRF .monash_score > 0)/len(monash_scoreRF ),
sum(monash_scoreRF .monash_score), sum(monash_scoreRF .monash_score)/len(monash_scoreRF ), np.std(monash_scoreRF .monash_score))