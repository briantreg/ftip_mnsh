import pandas as pd

dfAllGamesPlayers = pd.read_pickle('data/dfAllGamesPlayers.pkl')
dfAllFantasyScores = pd.read_pickle('data/dfAllFantasyScores.pkl')
dfAllGamesPlayersScores = pd.merge(dfAllGamesPlayers,
      dfAllFantasyScores,
      on = ['year','season_round','fw_player_id'],
      how = 'left')

dfFantasyAverage = dfAllGamesPlayersScores.groupby(by = ['fw_game_id','team'], as_index = False)['SupercoachPrice', 'AFLFantasyPrice'].mean()

pd.to_pickle(dfFantasyAverage, 'data/dfFantasyAverage.pkl')