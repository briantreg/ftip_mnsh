from random import seed
from random import randrange
from itertools import product
import pandas as pd
import numpy as np
import math



def eloMonashScore(eloScored, colScore, colWin):
    eloScored['monash_scores'] = ((1 + np.log2(eloScored[colScore])) * eloScored[colWin]) + (1 + np.log2(1 - eloScored[colScore])) * (1 - eloScored[colWin])
    return eloScored

def eloProbWin(elo_t, elo_o, flagHome, HGA, m):
    if flagHome == 0:
        HGA = - HGA
    nume2 = - (elo_t - elo_o + HGA)
    sum1 = nume2 / m 
    p1 = 1 / (math.pow(10, sum1) + 1)
    return p1

def eloNew(eloWin, eloLose, K, flagHome, HGA, m):
    p1 = eloProbWin(eloWin, eloLose, flagHome, HGA, m)
    p0 = 1 - p1
    eloWinNew = eloWin + K * (1 - p1) #Change for winner
    eloLoseNew = eloLose + K * (0 - p0) #Change for loser
    return eloWinNew, eloLoseNew

def eloCreate(data, colTeam, colOpponent, colHomeGame, colWin, eloInitialScore, HGA, K, m, colEloOut, colSeason, regressParam ):
    eloEloColProbNm = colEloOut
    eloEloColTeamNm = 'eloTeam'
    eloEloColOppntNm = 'eloOppnt'
    
    data[eloEloColProbNm] = 0.5
    data[eloEloColTeamNm] = eloInitialScore
    data[eloEloColOppntNm] = eloInitialScore
    eloCol = 'elo'
    eloCurrentScore = pd.DataFrame(
            {'team': pd.Series(data .team.unique()),
             eloCol: eloInitialScore})
    
    data  = data.sort_values(by  = ['date'],
                       ascending = [True]).reset_index(drop = True)
    n = data.shape[0]
    
    seasonCurrent = 0
    
    #start of loop
    for i in range(0, n):
        x = data.iloc[i] #Get the game to evaluate

        if not(x[colSeason] == seasonCurrent): #The game is from a new season
            league_mean = sum(eloCurrentScore[eloCol])/len(eloCurrentScore[eloCol])
            eloCurrentScore[eloCol] = eloCurrentScore[eloCol] + ((league_mean - eloCurrentScore[eloCol] ) * regressParam)
            seasonCurrent = x[colSeason]
            
        team = x[colTeam] #Team in the game
        opponent = x[colOpponent] #Opponent in the game
        teamHome = x[colHomeGame]
        
        eloTeamCurrent = eloCurrentScore[eloCurrentScore['team'] == team][eloCol].values[0]  #Current ELO for the team
        eloOpponentCurrent = eloCurrentScore[eloCurrentScore['team'] == opponent][eloCol].values[0]  #Current ELO for the opponent
        
        homeOpponent = 1 - teamHome
    
        p1 = eloProbWin(eloTeamCurrent, eloOpponentCurrent, teamHome, HGA, m) #Probability team will win
            
        if(x[[colWin]].values == 1): #New elos if team won
            eloTeamNew, eloOpponentNew = eloNew(eloTeamCurrent, eloOpponentCurrent, K, teamHome, HGA, m)
        if(x[[colWin]].values == 0): #New elos if team won
            eloOpponentNew, eloTeamNew = eloNew(eloOpponentCurrent, eloTeamCurrent, K, homeOpponent, HGA, m)
            
        eloCurrentScore.loc[eloCurrentScore['team'] == team, eloCol] = eloTeamNew #Assigning new team ELO back to ELO frame
        eloCurrentScore.loc[eloCurrentScore['team'] == opponent, eloCol] = eloOpponentNew #Assigning new opponent ELO back to ELO frame
        
        data.loc[i, eloEloColProbNm] = p1
        data.loc[i, eloEloColTeamNm] = eloTeamCurrent
        data.loc[i, eloEloColOppntNm] = eloOpponentCurrent
    
    return data


def eloScoreLogLoss(data, colWin, colElo):
    logloss_array =  (np.log(data[colElo]).values * data[colWin].values)  + (np.log(1 - data[colElo]).values * (1 - data[colWin].values))
    return np.mean(logloss_array)

# Split a dataset into k folds
def kfoldSplit(data, folds=3):
	data_split = list()
	data_copy = list(data.index)
	fold_size = int(len(data) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split

def gridExpand(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

def appendGameN(data, colGameN = 'gameN'):
        data[colGameN] = list(range(len(data)))
        return data
    
def eloGridSearch(data, k, eloParams, scoreFunc, excludeFirstN, colTeam = 'team', colOpponent = 'opponent', colHomeGame = 'home_game', colWin = 'model_target', colEloOut = 'eloProb', colSeason = 'year',     test_ids_file = 'test_fw_ids.pkl'):
     
    kfoldGrid = gridExpand(eloParams) 

    test_ids = pd.read_pickle(test_ids_file)
    testIndex = pd.DataFrame(data[data.fw_game_id.isin(test_ids.fw_game_id.values)]).index

    colGameN = 'gameN'
    dataDatesTeam = data[['team', 'date', 'fw_game_id']]
    dataDataOpponent = data[['opponent', 'date', 'fw_game_id']] 
    dataDataOpponent.rename(columns = {'opponent': 'team'}, inplace = True)
    dataDates = dataDatesTeam.append(dataDataOpponent).reset_index(drop = True)
    dataDates = dataDates.sort_values('date')
    dataDates = dataDates.groupby('team').apply(appendGameN, colGameN = colGameN)    
    dataExcludeIds = dataDates[dataDates[colGameN] <= excludeFirstN]['fw_game_id'].unique()
    dataXval  = data[~data['fw_game_id'].isin(dataExcludeIds)]
    dataXval = dataXval[~dataXval.index.isin(testIndex)]
    kfoldSplits = kfoldSplit(dataXval, k)

    gridResultsColumns = ['scoreMean','scoreMax','scoreMin']
    gridResultsColumns.extend(list(kfoldGrid.columns))
    gridResults = pd.DataFrame(columns = gridResultsColumns)

    for row in kfoldGrid.itertuples(index = False):
        
        HGA = row.HGA
        K = row.K
        eloInitialScore = row.eloInitialScore
        m = row.m
        regressParam = row.regressParam
        rowDict = {'HGA': HGA,
                   'K': K,
                   'eloInitialScore':eloInitialScore,
                   'm': m,
                   'regressParam': regressParam}
        
        
        scoreList = list()
        
        dataScore = eloCreate(data, colTeam, colOpponent, colHomeGame, colWin, eloInitialScore, HGA, K, m, colEloOut, colSeason, regressParam)
        
        for fold in range(k):
            
            dataTest = dataScore[dataScore.index.isin(kfoldSplits[fold])]
            scoreRound = scoreFunc(dataTest, colWin, colEloOut)
            scoreList.append(scoreRound)
                        
        scoreDict = {'scoreMean' : sum(scoreList)/len(scoreList),
                     'scoreMax' : max(scoreList),
                     'scoreMin' : min(scoreList)}
        
        #paramsRow = pd.Series(row, index = row._fields).to_dict        
        paramResult = {**scoreDict, **rowDict}
        #paramResult = scoreDict 
        #print(paramsRow)
        print(paramResult)
        
        paramResult = pd.Series(data = paramResult, index = paramResult.keys())
        
        gridResults = gridResults.append(paramResult, ignore_index = True)
        
    gridResults = gridResults.sort_values(by = ['scoreMean'], axis = 0, ascending = False).reset_index(drop = True)
    return gridResults 