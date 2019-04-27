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

data = afl_DF
colTeam = 'team'
colOpponent = 'opponent'
colHomeGame = 'home_game'
colWin = 'model_target'
colSeason = 'year'
eloInitialScore = 1500
HGA = 40
K = 100
m = 400
colEloOut = 'eloProb'
regressParam = 0.1

afl_ELO = eloCreate(data, colTeam, colOpponent, colHomeGame, colWin, eloInitialScore, HGA, K, m, colEloOut, colSeason, regressParam)

def eloScoreLogLoss(data, colWin, colElo):
    logloss_array =  (np.log(data[colElo]).values * data[colWin].values)  + (np.log(1 - data[colElo]).values * (1 - data[colWin].values))
    return np.mean(logloss_array)

eloScoreLogLoss(afl_ELO , colWin, colEloOut)