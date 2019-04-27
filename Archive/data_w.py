#%%
import os

os.chdir("C:\\Users\\Brian\\Desktop\\local_git\\ftip_mnsh")

#%%
import utilities as utl
import pandas as pd
import numpy as np


#%%
# =============================================================================
# Initial load and clean data
# =============================================================================
data_file = 'afl_DF.pkl'
first_year = 2012 #earliest year to include in data wrangling. Zero means, all of them.
exclude_finals = 1 #1 = exclude, 0 = include

afl_DF, dropped_cols = dataLoadClean(data_file, first_year, exclude_finals)

#%%
# =============================================================================
# # Impute new features with scraped data
# =============================================================================

afl_DF = calcScoreMarginWin(afl_DF)
#%%

afl_DF = appendModelTarget(afl_DF, 'Win')

#%%
# =============================================================================
# Create an 
# =============================================================================


afl_DF = createIndex('round_index', afl_DF, ['year', 'season_round'], [True, True])
afl_DF = createIndex('game_index', afl_DF, ['date','fw_game_id'], [True, True])

#%%

# =============================================================================
# Remove features that are not wanted for the model
# =============================================================================
###Statistics to remove from the data
cols_to_drop= ['Kicks', 'Handballs','Marks', 'Tackles', 'Hitouts', 'Frees_For', 'Frees_Against',
       'Goals_Kicked', 'Behinds_Kicked', 'Rushed_Behinds', 'Disposals_Per_Goal', 'Clearances',
       'Clangers', 'In50s_Per_Scoring_Shot', 'Inside_50s_Per_Goal',  'In50s_Goal',
       'Contested_Possessions', 'Uncontested_Possessions', 'Effective_Disposals', 'Contested_Marks',
        'One_Percenters', 'Bounces', 'Turnovers', 'Intercepts', 'Tackles_Inside_50', 'Kick_to_Handball_Ratio']


afl_DF = dropStatCols(afl_DF, cols_to_drop)

# =============================================================================
# Create a column representing the % of each 'stat' achieved by "team"
# =============================================================================

pcnt_naming = 'Pcnt'
afl_DF = calcStatPercentage(afl_DF, pcnt_naming)
# =============================================================================
# Remove all of the columns that are include "Team" or "Oppnt"
# =============================================================================

afl_DF = dropRawStats(afl_DF)

#%%
# =============================================================================
# Columns to include in creating "average" values
# =============================================================================

mean_N = [20]
cols_to_apply = [col for col in afl_DF.columns if pcnt_naming in col] #All columns with pcnt are included
cols_to_apply.extend(['Win','Margin']) #Additional columns to include

afl_DF = calcStatMeans(afl_DF, mean_N, cols_to_apply)

afl_DF = dropCols(afl_DF, cols_to_apply) #Remove columns used to create means
    
 #%%

afl_DF = appendTeamOpponentMeans(afl_DF)
#%%

# =============================================================================
# Create ratios from the means
# =============================================================================


afl_DF = createMeanRatio(afl_DF)
# =============================================================================
# Difference the means (usually where a ratio would cause issues with inf)
# =============================================================================

afl_DF = createMeanDiff(afl_DF)
# =============================================================================
# Remove all opponent and team average columns
# =============================================================================

afl_DF = dropNormalMeanCols(afl_DF)
# =============================================================================
# Remove draws
# =============================================================================

afl_DF = dropDraws(afl_DF)

# =============================================================================
# Create probabilities from the odds for benchmarking
# =============================================================================


afl_DF = createOddsProb_MonashScore(afl_DF)
# =============================================================================

# Decide what sort of game mix should be kept for training. All of the home or all away, a mix of both or all of the games
# =============================================================================

games_to_keep = 'mix' #'mix' = mix of home and away, 'home' 'away' = home and away respectively. Else
afl_DF = createGameMix(afl_DF, games_to_keep)

afl_CLEAN = afl_DF
# =============================================================================
# Drop columns that are not wanted in the ML data
# =============================================================================
drop_cols = ['season_round','round_char','date','team','opponent', 'location','attendance','final',
             'team_odds','oppnt_odds','team_line','oppnt_line', 'Margin','Result','Margin_ELO_m8','game_index',
             'oppnt_oddsprob', 'team_oddsprob ', 'team_oddsprobwght', 'monash_scorewin', 'monash_scorelloss','monash_score']

afl_ML = createMLData(afl_CLEAN, drop_cols)