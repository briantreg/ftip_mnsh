import pandas as pd
import numpy as np

from boruta import BorutaPy

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def prepareMLData(data, test_ids_file, year_weights):
    afl_ML = data
    del data
    test_ids = pd.read_pickle(test_ids_file)
    afl_ML = afl_ML.dropna(axis = 1)
    
    train = afl_ML[~(afl_ML.fw_game_id.isin(test_ids.fw_game_id.values))]
    test = afl_ML[afl_ML.fw_game_id.isin(test_ids.fw_game_id.values)]
    
    target = 'Win'
    M_cols = ['fw_game_id', 'year', 'round_index']
    
    M_train = train[M_cols]
    M_test = test[M_cols]
    sample_weights = M_train.year.map(year_weights)
    
    Y_train = train[target]
    Y_test = test[target]
    
    M_cols.append(target)
    X_cols = [col for col in afl_ML.columns if col not in (M_cols)]
    
    X_train = train[X_cols]
    X_test = test[X_cols]
    
    train_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = train_scaler.transform(X_train)
    X_test_scaled = train_scaler.transform(X_test)
    Y_trainval = Y_train.values
    return X_test, X_test_scaled, X_train, X_train_scaled, Y_test, Y_train, Y_trainval, M_train, M_test, target, sample_weights

def selBoruta(X_test, X_test_scaled, X_train, X_train_scaled, Y_trainval):

    #n_jobs=-1, class_weight='balanced', max_depth=5,n_estimators='auto', verbose=2, random_state=1, rseed = 80085
    n_jobs=-1
    class_weight='balanced'
    max_depth=5
    n_estimators='auto'
    verbose=2
    random_state=1
    rseed = 80085
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=n_jobs, class_weight=class_weight, max_depth=max_depth)
    
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators=n_estimators, verbose=verbose, random_state=random_state)
    np.random.seed(rseed)
    feat_selector.fit(X_train_scaled, Y_trainval)
    
    criteria = pd.Series(feat_selector.support_)
    X_train_boruta = feat_selector.transform(X_train_scaled)
    X_train_boruta = pd.DataFrame(X_train_boruta, index = X_train.index, columns = X_train.columns[criteria].values)
    
    criteria = pd.Series(feat_selector.support_)
    X_test_boruta = feat_selector.transform(X_test_scaled)
    X_test_boruta = pd.DataFrame(X_test_boruta, index = X_test.index, columns = X_test.columns[criteria].values)
    
    return X_test_boruta, X_train_boruta

def trainSVC(X_train_boruta, Y_train, param_set, score, sample_weights):
    SVC_clf = GridSearchCV(svm.SVC(probability = True),
            param_grid = param_set,
            scoring = score,
            cv = 5,
            verbose = 3)
    SVC_clf.fit(X_train_boruta, Y_train, sample_weight = sample_weights.values)
    return SVC_clf

def trainRF(X_train_boruta, Y_train, param_set, score, sample_weights):
    RF_clf = GridSearchCV(RandomForestClassifier(),
            param_grid = param_set,
            scoring = score,
            cv = 5,
            verbose = 3)
    RF_clf.fit(X_train_boruta, Y_train, sample_weight = sample_weights.values)
    return RF_clf 


def trainRandRF(X_train_boruta, Y_train, param_set, score, sample_weights, n_iter):
    RFrand_clf = RandomizedSearchCV(RandomForestClassifier(),
            n_iter = n_iter,
            param_distributions = param_set,
            scoring = score,
            cv = 5,
            verbose = 3)
    RFrand_clf.fit(X_train_boruta, Y_train)#, sample_weight = sample_weights.values)
    return RFrand_clf


def scoreMonash(classifier, X_test_data, Y_test_data):
    Y_test_probabilities = classifier.predict_proba(X_test_data)
    Y_test_probabilities = pd.DataFrame(Y_test_probabilities, index = Y_test_data.index)
    Y_test_probabilities = Y_test_probabilities.join(Y_test_data)
    Y_test_probabilities['Win']
    Y_test_probabilities['monash_score'] = ((1 + np.log2(Y_test_probabilities.iloc[:,1])) * Y_test_probabilities['Win']) + (1 + np.log2(1 - Y_test_probabilities.iloc[:,1])) * (1 - Y_test_probabilities['Win'])
    return Y_test_probabilities

def scoreOddsBenchmark(test_ids_file, data):
    test_ids = pd.read_pickle(test_ids_file)
    test = data[data.fw_game_id.isin(test_ids.fw_game_id.values)]
    return sum(test['monash_score'])



def calcStatMeans(data, mean_N ,cols):

    data = data.sort_values(by = ['team','game_index'],# Sort data by team and game_index for running mean
                 ascending = [True, True]
                     ).reset_index(drop=True)
    
    for N in mean_N:
        for col in cols:
            data = data.groupby('team').apply(newMeanCol, col_nm = col, N = N)
    return data[~(data.isna().any(axis = 1))] #Remove na created in the process

def createOddsProb_MonashScore(data):
    oppnt_odds = data[['oppnt_odds']]
    oppnt_oddsprob = 1 / (oppnt_odds + 0.000000000001)
    data['oppnt_oddsprob'] = oppnt_oddsprob
    team_odds = data[['team_odds']]
    team_oddsprob = 1 / (team_odds + 0.000000000001)
    data['team_oddsprob '] = team_oddsprob
    data['team_oddsprobwght'] = (data['team_oddsprob '] + (1 - data['oppnt_oddsprob']) ) / 2
    data['monash_scorewin'] = (1 + np.log2(pd.to_numeric(data['team_oddsprobwght']) ) ) * data['model_target']
    data['monash_scorelloss'] = (1 + np.log2(1 - pd.to_numeric(data['team_oddsprobwght']) ) ) * (1 - data['model_target'])
    data['monash_score'] = afl_DF['monash_scorewin'] + data['monash_scorelloss']
    
    return data