import utilities as utl
import pandas as pd
import numpy as np
import datetime as dt

teams = ['Carlton','Collingwood','Melbourne', 'Adelaide', 'Brisbane', 'Western Bulldogs',  'St Kilda', 'GWS', 'Fremantle']
opponents = ['Richmond','Geelong','Port Adelaide', 'Hawthorn', 'West Coast','Sydney', 'Gold Coast', 'Essendon', 'North Melbourne']
year = 2019
round_n = 1
games = range(1,len(teams)+1)

def dataWrangle(first_year, exclude_finals, games_to_keep, data_file):
    ##Initial load and clean data
    afl_DF_main = pd.read_pickle(data_file) #Read in the data
    afl_NEW = pd.DataFrame({'team':teams, 
              'opponent':opponents, 
              'home_game': 1, 
              'Result': 1, 
              'fw_game_id': games, 
              'date':dt.datetime(2099, 12, 31), 
              'season_round': round_n, 
              'round_char': str(round_n),
              'year': year})
    afl_NEW = pd.concat([afl_DF_main[afl_DF_main['team'] ==  'none'], afl_NEW], sort = False)
    afl_NEW = afl_NEW.fillna(0.0)
    afl_DF = pd.concat([afl_DF_main, afl_NEW]) 
    
    afl_DF_main = afl_DF_main[afl_DF_main.year >= first_year].reset_index() #Filter the data toonly include data from years greater than or equal to <first_year>
    afl_DF = afl_DF_main.loc[0:,~(afl_DF_main.isna().any())] #Remove any columns that have NAs in the columns
    afl_DF.columns.duplicated()
    if exclude_finals == 1: #Drop all finals games when <exclude_finals> is 1
        afl_DF = afl_DF[afl_DF_main['final'] == 0]
    
    ###Create an index for each of the rounds
    round_index = afl_DF.loc[0:][['year','season_round']] #New dataset with year and season_round
    round_index = round_index.drop_duplicates(subset = ['year', 'season_round']) #Remove duplicate round rows
    round_index = round_index.sort_values(by = ['year', 'season_round'], #Order the rounds by year and round, reset the index
                           ascending = [True, True]).reset_index(drop=True) 
    round_index['round_index'] = pd.DataFrame(list(round_index.index)) #Create a new column with the index
    afl_DF = pd.merge(afl_DF, #Join the index back to the main data
            round_index,
            left_on = ['year','season_round'],
            right_on = ['year','season_round'],
            how = 'left')
    
    ###Create an index for each of the games
    game_index = afl_DF.loc[0:][['fw_game_id','date']] #New dataset with fw_game_id and date
    game_index = game_index.drop_duplicates(subset = ['fw_game_id','date']) #Remove duplicate game rows
    game_index = game_index.sort_values(by = ['date', 'fw_game_id'], #Order the games by date and fw_game_id, reset the index
                           ascending = [True, True]
                                       ).reset_index(drop=True)
    game_index['game_index'] = pd.DataFrame(list(game_index.index)) #Create a new column with the index
    afl_DF = pd.merge(afl_DF, #Join the index back to the main data
            game_index,
            left_on = ['fw_game_id','date'],
            right_on = ['fw_game_id','date'],
            how = 'left')
    
    ###Sort data by team and game_index for running mean
    afl_DF = afl_DF.sort_values(by = ['team','game_index'],
                 ascending = [True, True]
                     ).reset_index(drop=True)
    
    ###Create new score, margin, result and win variables
    afl_DF['ScoreTeam'] = (afl_DF['Goals_KickedTeam'] * 6) + afl_DF['Behinds_KickedTeam'] #Goals for the team * 6 plus behinds for the team is the score 
    afl_DF['ScoreOppnt'] = (afl_DF['Goals_KickedOppnt'] * 6) + afl_DF['Behinds_KickedOppnt'] #Goals for the opponent * 6 plus behinds for the opponent is the score 
    afl_DF['Margin'] = afl_DF['ScoreTeam'] - afl_DF['ScoreOppnt'] #The final margin calculated as the score for the team minus the score for the opponent 
    
    afl_DF.loc[afl_DF['Margin'] == 0, 'Result'] = 0 #When margin is 0, 0
    afl_DF.loc[afl_DF['Margin'] > 0, 'Result'] = 1 #When margin > 0, 1
    afl_DF.loc[afl_DF['Margin'] < 0, 'Result'] = -1 #When margin < 0, -1
    
    afl_DF.loc[:, 'Win'] = 0 #Make win a zero
    afl_DF.loc[afl_DF['Margin'] > 0, 'Win'] = 1 #When margin > 0 it's a win = 1
    
    ###Statistics to remove from the data
    drop_stat_cols = ['Kicks', 'Handballs','Marks', 'Tackles', 'Hitouts', 'Frees_For', 'Frees_Against',
           'Goals_Kicked', 'Behinds_Kicked', 'Rushed_Behinds', 'Disposals_Per_Goal', 'Clearances',
           'Clangers', 'In50s_Per_Scoring_Shot', 'Inside_50s_Per_Goal',  'In50s_Goal',
           'Contested_Possessions', 'Uncontested_Possessions', 'Effective_Disposals', 'Contested_Marks',
            'One_Percenters', 'Bounces', 'Turnovers', 'Intercepts', 'Tackles_Inside_50', 'Kick_to_Handball_Ratio']
    drop_stat_cols_fin = [string + 'Team' for string in drop_stat_cols]
    drop_stat_cols_fin.extend([string + 'Oppnt' for string in drop_stat_cols_fin])
    drop_stat_cols_fin = [col for col in drop_stat_cols_fin if col in afl_DF.columns]
    
    #Need to check if this is doing anything
    afl_DF = afl_DF.loc[:,~afl_DF.columns.duplicated()]
    
    ####Create a column representing the % of each 'stat' achieved by "team"
    
    team_array =  utl.colLookup(afl_DF, 'Team').values #get all columns with "Team" in the name, then return the cell values
    oppnt_array =  utl.colLookup(afl_DF, 'Oppnt').values #get all columns with "Opponent" in the name, then return the cell values
    diff_array = team_array / (team_array + oppnt_array + 0.00001) #Create an array of values which is the % of the stat belonging to team. Note: adding small positive value to avoid inf
    afl_DF[list( utl.colLookup(afl_DF, 'Team').columns.str.replace('Team','Pcnt'))] = pd.DataFrame(diff_array) #Create a new set of columns and insert the array of values back into main dataframe
    
    afl_DF = afl_DF[[col for col in afl_DF.columns if 'Team' not in col and 'Oppnt' not in col]]
    #afl_DF = afl_DF[[col for col in afl_DF.columns if 'Oppnt' not in col]]
    #afl_DF = afl_DF[[col for col in afl_DF.columns if 'Team' not in col]]
    
    ELO_col = ['Margin']
    ELO_mean_N = [8]
    for n in ELO_mean_N:
        for col in ELO_col:
            afl_DF = afl_DF.groupby('team').apply( utl.newMeanCol, col_nm = col, N = n)
    afl_DF =  utl.NANConsolidate(afl_DF, 'fw_game_id')
    
    afl_DF.columns = afl_DF.columns.str.replace("_mean","_T_ELO_mean")
    
    cols = ['fw_game_id', 'team']
    cols.extend([col for col in afl_DF.columns if 'ELO' in col])
    elo_DF = afl_DF[cols]
    elo_DF.columns = elo_DF.columns.str.replace('T_ELO','O_ELO')
    elo_DF.columns = elo_DF.columns.str.replace('team','opponent')
    
    afl_DF = pd.merge(afl_DF,
             elo_DF,
             left_on = ['fw_game_id','opponent'],
             right_on = ['fw_game_id','opponent'],
            how = 'left')
    ELO_mean_N = [8]
    afl_DF = afl_DF[(afl_DF["round_index"] > (max(ELO_mean_N) + 2))]
    afl_DF['Margin_ELO_m8'] = pd.DataFrame(afl_DF.Margin - (afl_DF.Margin_T_ELO_mean8 - afl_DF.Margin_O_ELO_mean8))
    afl_DF = afl_DF.drop(['Margin_T_ELO_mean8','Margin_O_ELO_mean8'], axis = 1)
    
    cols_to_apply = [col for col in afl_DF.columns if 'Pcnt' in col]
    cols_to_apply.extend([col for col in afl_DF.columns if 'ELO' in col])
    cols_to_apply.extend(['Win','Margin'])
    
    mean_N = [20]
    for N in mean_N:
        for col in cols_to_apply:
            afl_DF = afl_DF.groupby('team').apply( utl.newMeanCol, col_nm = col, N = N)
    afl_DF =  utl.NANConsolidate(afl_DF, 'fw_game_id')
    
    afl_DF = afl_DF[~(afl_DF.isna().any(axis = 1))]
    
    afl_DF = afl_DF[[col for col in afl_DF.columns if 'Pcnt' not in col or 'mean' in col]]
    
    feature_col = ['fw_game_id','team']
    feature_col.extend([col for col in afl_DF.columns if 'mean' in col])
    afl_DF_oppnt = afl_DF[feature_col]
    afl_DF_oppnt.columns = afl_DF_oppnt.columns.str.replace('team','opponent')
    afl_DF_oppnt.columns = afl_DF_oppnt.columns.str.replace('mean','O_mean')
    afl_DF.columns = afl_DF.columns.str.replace('mean','T_mean')
    
    team_pcnt_col = [col for col in afl_DF.columns if 'mean' in col]
    oppnt_pcnt_col = [col for col in afl_DF_oppnt.columns if 'mean' in col]
    team_pcnt_col = [col for col in team_pcnt_col if 'Win_' not in col and 'Margin_' not in col]
    oppnt_pcnt_col = [col for col in oppnt_pcnt_col if 'Win_' not in col and 'Margin_' not in col]
    
    team_other_col = [col for col in afl_DF.columns if ('Win_' in col or 'Margin_' in col) and '_T_' in col]
    oppnt_other_col = [col for col in afl_DF_oppnt.columns if ('Win_' in col or 'Margin_' in col) and '_O_' in col]
    
    afl_DF = pd.merge(afl_DF,
                     afl_DF_oppnt,
                     left_on = ['fw_game_id','opponent'],
                     right_on = ['fw_game_id','opponent'],
                     how='left')
    
    team_pcnt_mean = afl_DF[team_pcnt_col].values
    oppnt_pcnt_mean = afl_DF[oppnt_pcnt_col].values
    diff_pcnt_mean = team_pcnt_mean / (team_pcnt_mean + oppnt_pcnt_mean + 0.000001)
    
    team_other_mean = afl_DF[team_other_col].values
    oppnt_other_mean = afl_DF[oppnt_other_col].values
    diff_other_mean = team_other_mean - oppnt_other_mean
    
    afl_DF[[col.replace('T','D') for col in team_pcnt_col]] = pd.DataFrame(diff_pcnt_mean)
    afl_DF[[col.replace('T','D') for col in team_other_col]] = pd.DataFrame(diff_other_mean)
    
    afl_DF = afl_DF[[col for col in afl_DF.columns if '_T_' not in col and '_O_' not in col]]
    
    afl_DF = afl_DF[~(afl_DF.Result == 0)]
    
    oppnt_odds = afl_DF[['oppnt_odds']]
    oppnt_oddsprob = 1 / (oppnt_odds + 0.000000000001)
    afl_DF['oppnt_oddsprob'] = oppnt_oddsprob
    team_odds = afl_DF[['team_odds']]
    team_oddsprob = 1 / (team_odds + 0.000000000001)
    afl_DF['team_oddsprob '] = team_oddsprob
    afl_DF['team_oddsprobwght'] = (afl_DF['team_oddsprob '] + (1 - afl_DF['oppnt_oddsprob']) ) / 2
    afl_DF['monash_scorewin'] = (1 + np.log2(pd.to_numeric(afl_DF['team_oddsprobwght']) ) ) * afl_DF['Win']
    afl_DF['monash_scorelloss'] = (1 + np.log2(1 - pd.to_numeric(afl_DF['team_oddsprobwght']) ) ) * (1 - afl_DF['Win'])
    afl_DF['monash_score'] = afl_DF['monash_scorewin'] + afl_DF['monash_scorelloss']
    
    afl_NEW = afl_DF[afl_DF['fw_game_id'] in range(1,10)]
    afl_DF = afl_DF[afl_DF['fw_game_id'] not in range(1,10)]
    
    if games_to_keep not in ['mix','home','away','all']:
        print('error, <games_to_keep> variable not set correctly')
    elif games_to_keep == 'mix':
        odds = [x for x in afl_DF.fw_game_id if int(x) % 2 > 0]
        evens = [x for x in afl_DF.fw_game_id if int(x) % 2 == 0]
        a = afl_DF[afl_DF.fw_game_id.isin(odds)]
        b = afl_DF[afl_DF.fw_game_id.isin(evens)]
        a = a[a.home_game == 0] 
        b = b[b.home_game == 1] 
        afl_DF = pd.concat([a,b])
        del a, b
    elif games_to_keep == 'home':
        afl_DF = afl_DF[afl_DF.home_game == 1]
    elif games_to_keep == 'away':
        afl_DF = afl_DF[afl_DF.home_game == 0]
    
    drop_cols = ['season_round','round_char','date','team','opponent', 'location','attendance','final',
                 'team_odds','oppnt_odds','team_line','oppnt_line', 'Margin','Result','Margin_ELO_m8','game_index',
                 'oppnt_oddsprob', 'team_oddsprob ', 'team_oddsprobwght', 'monash_scorewin', 'monash_scorelloss','monash_score']
    drop_cols_fin = [col for col in drop_cols if col in afl_DF.columns]
    afl_ML = afl_DF.drop(drop_cols_fin, axis = 1)
    
    cols = list(afl_ML.columns)
    cols.remove('Win')
    cols.append('Win')
    afl_ML = afl_ML [cols]
        
    return afl_DF, afl_ML, afl_NEW