import pandas as pd
import numpy as np

from boruta import BorutaPy

from sklearn.model_selection import GridSearchCV
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

def selBoruta(X_test, X_train, X_train_scaled, Y_trainval, n_jobs=-1, class_weight='balanced', max_depth=5,n_estimators='auto', verbose=2, random_state=1, rseed = 80085)
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
def 
    Y_test_predictions = SVC_clf.predict(X_test_boruta)
    Y_test_probabilities = SVC_clf.predict_proba(X_test_boruta)
    Y_test_predictions = pd.DataFrame(Y_test_predictions, index = Y_test.index, columns = ['predict'])
    Y_test_probabilities = pd.DataFrame(Y_test_probabilities, index = Y_test.index)
    Y_test_outcomes = Y_test_probabilities.join(Y_test_predictions)
    Y_test_outcomes = Y_test_outcomes.join(Y_test)
    s = abs(Y_test_outcomes.iloc[0:,2] - Y_test_outcomes.iloc[0:,3])
    1 - sum(s)/len(s)
    
    afl_DF.loc[afl_DF['Margin'] == 0, 'Result'] = 0
    afl_DF.loc[afl_DF['Margin'] > 0, 'Result'] = 1
    afl_DF.loc[afl_DF['Margin'] < 0, 'Result'] = -1
    
    prediction_set = Y_test_probabilities.join(M_test).join(X_test.home_game)
    prediction_set = prediction_set.drop(0,axis=1)
    prediction_set = pd.merge(prediction_set.loc[prediction_set['home_game'] == 1,[1,'fw_game_id','year','round_index','home_game']],
             prediction_set.loc[prediction_set['home_game'] == 0, [1,'fw_game_id']],
             left_on = ['fw_game_id'],
             right_on = ['fw_game_id'],
            how = 'left')
    prediction_set[['fin_pred']] = pd.DataFrame((1-prediction_set[['1_y']].values+prediction_set[['1_x']].values)/2)
    prediction_set
    
    prediction_set.loc[prediction_set['home_game'] == 0, [1,'fw_game_id','home_join']]
    
    SVC_clf.best_estimator_
    
    Y_test_outcomes.loc[Y_test_outcomes.predict == Y_test_outcomes.Win,'score'] = Y_test_outcomes[Y_test_outcomes.predict == Y_test_outcomes.Win].iloc[0:,0:2].max(axis = 1).values
    
    Y_test_outcomes.loc[~(Y_test_outcomes.predict == Y_test_outcomes.Win),'score'] = Y_test_outcomes[~(Y_test_outcomes.predict == Y_test_outcomes.Win)].iloc[0:,0:2].min(axis = 1).values
    
    Y_test_outcomes.loc[0:,]
    
    Y_test_outcomes['points'] = 1 + np.log2(Y_test_outcomes['score'])
    sum(Y_test_outcomes['points']) / len(Y_test_outcomes['points']) * 207
    
    SVC_clf.best_estimator_
    
    SVC_clf.grid_scores_
    
