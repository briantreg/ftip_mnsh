import utilities.utilitiesElo as utlElo
import dataWrangling as dw
import pandas as pd
eloParams = {'HGA': [40],
               'K': [30,40,50],
               'm': [400],
               'eloInitialScore': [1500],
               'regressParam': [0.5],
               'p': [0.04],
               'KMargin': [0.2,0.3,0.5]}

waste, afl_DF = dw.dataWrangling(new_data = '', outputType = 'none')
del waste

results = utlElo.eloGridSearch(afl_DF, 5, eloParams, utlElo.eloScoreLogLoss, 20)
a = ['t','w','s']
a.extend(['b','x'])
a.remove('t')
afl_DF['Margin']

waste, afl_DF = dw.dataWrangling(new_data = '', outputType = 'none')
del waste


dataScored = utlElo.eloMonashScore(utlElo.dataScored, colScore = 'eloProb', colWin = 'model_target')

test_ids = pd.read_pickle('test_fw_ids.pkl')
testIndex = pd.DataFrame(dataScored[dataScored.fw_game_id.isin(test_ids.fw_game_id.values)]).index
dataScoredTest = dataScored.loc[testIndex]
sum(dataScoredTest.monash_scores)/len(dataScoredTest.monash_scores)*207
22*9