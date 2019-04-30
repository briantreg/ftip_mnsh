year_start = 2013
year_end = 2019

import urllib.request
#from bs4 import BeautifulSoup
import pandas as pd
#import datetime
import re
import os
#import sys
import gc

import utilities.utilitiesScrape as utlScrape

###Get all the game ids in the range
game_ids = []
for year in range(year_start,year_end + 1):
    try:
        afl_year_url = 'https://www.footywire.com/afl/footy/ft_match_list?year=' + str(year)
        soup = utlScrape.getSoup(afl_year_url)
        for a in soup.find_all("a", href=re.compile(r"^ft_match_statistics")):
            game_ids.append(
                int(a['href'].replace('ft_match_statistics?mid=',''))
            )
    except urllib.error.URLError:
        print('No internet connection')
        break

if os.path.isfile('./data/dfAllGamesPlayers.pkl'):
    dfAllGamesPlayers_read = pd.read_pickle('data/dfAllGamesPlayers.pkl')

if 'dfAllGamesPlayers_read' in locals():
    scraped_ids = list(dfAllGamesPlayers_read['fw_game_id'].astype('int'))
    game_ids = [x for x in game_ids if x not in scraped_ids]

for fw_game_id in game_ids:
    
    print(fw_game_id)
    
    aflGameURL = 'https://www.footywire.com/afl/footy/ft_match_statistics?mid=' + str(fw_game_id)
    
    ###For each game get player data
    soup = utlScrape.getSoup(aflGameURL)
    
    #######NOT DOING ANYTHING BUT READY TO GO
    htablePlayersTeam = soup.find('table', border="0", cellpadding="3", cellspacing="0", width="823") #find the first table with thinfo
    htablePlayersOpponent = htablePlayersTeam.find_next('table', border="0", cellpadding="3", cellspacing="0", width="823")
    
    htablePlayersTeam = utlScrape.prettyPlayerStatTable(htablePlayersTeam)
    htablePlayersOpponent = utlScrape.prettyPlayerStatTable(htablePlayersOpponent)
    
    ###For each game get player data
    hrefsPlayers = soup.find_all('td', align='left', height = '18')
    
    for i  in range(len(hrefsPlayers)):
        
        PlayerLink = hrefsPlayers[i].a
        PlayerLink = [[PlayerLink.text, PlayerLink['href']]]
    
        try:
            tablePlayerLink 
        except NameError:
            tablePlayerLink = pd.DataFrame(PlayerLink, columns = ['Player','fw_player_id'])
        else:
            newPlayerLink = pd.DataFrame(PlayerLink, columns = ['Player','fw_player_id'])
            tablePlayerLink = tablePlayerLink.append(newPlayerLink, ignore_index = True)
            
    bsc_stat_tbl = utlScrape.getStatTable(soup)
    home_team = bsc_stat_tbl.iloc[1,1]
    away_team = bsc_stat_tbl.iloc[1,3]
    year, round_char, season_round = utlScrape.scrapeRound(soup)
        
    htablePlayersTeam = utlScrape.appendGameDetails(htablePlayersTeam, tablePlayerLink, year, round_char, season_round, fw_game_id)
    htablePlayersTeam['team'] = home_team
    htablePlayersOpponent = utlScrape.appendGameDetails(htablePlayersOpponent, tablePlayerLink, year, round_char, season_round, fw_game_id)
    htablePlayersOpponent['team'] = away_team
    tableGamePlayers = pd.concat([htablePlayersOpponent, htablePlayersTeam])
    
    try: 
        dfAllGamesPlayers 
    except NameError:
        dfAllGamesPlayers = tableGamePlayers
    else: 
        dfAllGamesPlayers = pd.concat([dfAllGamesPlayers, tableGamePlayers])

    try: 
        dfAllPlayersLink 
    except NameError:
        dfAllPlayersLink = tablePlayerLink
    else: 
        dfAllPlayersLink = pd.concat([dfAllPlayersLink, tablePlayerLink])
    dfAllPlayersLink = dfAllPlayersLink.drop_duplicates()
    dfAllGamesPlayers = dfAllGamesPlayers.drop_duplicates()
    gc.collect()

dfAllGamesPlayers = dfAllGamesPlayers[['year','round_char','season_round','fw_game_id','fw_player_id','team']]
if os.path.isfile('./data/dfAllGamesPlayers.pkl'):
    dfAllGamesPlayers_read = pd.read_pickle('data/dfAllGamesPlayers.pkl')
    dfAllGamesPlayers_read  = pd.concat([dfAllGamesPlayers_read, dfAllGamesPlayers])
    dfAllGamesPlayers_read  = dfAllGamesPlayers_read .drop_duplicates()
pd.to_pickle(dfAllGamesPlayers_read , "data/dfAllGamesPlayers.pkl")
del dfAllGamesPlayers_read 

if os.path.isfile('./data/dfAllPlayersLink.pkl'):
    dfAllPlayersLink_read = pd.read_pickle('data/dfAllPlayersLink.pkl')
    dfAllPlayersLink_read = pd.concat([dfAllPlayersLink_read, dfAllPlayersLink])
    dfAllPlayersLink_read = dfAllPlayersLink_read.drop_duplicates()
pd.to_pickle(dfAllPlayersLink_read, 'data/dfAllPlayersLink.pkl')   
del dfAllPlayersLink_read

##########Scraping Fantasy Values
for i in range(len(dfAllPlayersLink)):
    PlayerLink = dfAllPlayersLink.iloc[i]
    
    hrefAflFantasy = 'https://www.footywire.com/afl/footy/pr' + PlayerLink.fw_player_id[2:]
    hrefSupercoach = 'https://www.footywire.com/afl/footy/pu' + PlayerLink.fw_player_id[2:]
    
    FantasySoup = utlScrape.getSoup(hrefAflFantasy)
    
    first_season = 1
    r = 0
    
    while r < 1:
        if first_season == 1:
            htmlFantasy = FantasySoup.find("div", id=re.compile("player-fantasy-round-*"))
            first_season = 0
        elif first_season == 0:
            htmlFantasy = htmlFantasy.find_next("div", id=re.compile("player-fantasy-round-*"))
        
        try:
            htmlFantasy['id']
        except TypeError:
            r = 1
            continue
            
        year = htmlFantasy['id']
        year = int(year[21:])
        
        tableFantasy  = htmlFantasy.find('table',width = "688", border = "0")
        tableFantasy = pd.read_html(str(tableFantasy ))
        tableFantasy = tableFantasy[0]
        columnsFantasy = tableFantasy.iloc[0,:]
        tableFantasy = pd.DataFrame(tableFantasy.iloc[1:,:])
        tableFantasy.columns = columnsFantasy
        tableFantasy['year'] = year
        
        try: 
            tableAllFantasy 
        except NameError:
            tableAllFantasy = tableFantasy
        else: 
            tableAllFantasy = pd.concat([tableAllFantasy , tableFantasy])
    
    columnsAllFantasy = ['Round','AFLFantasyPrice','AFLFantasyScore', 'AFLFantasyValue','year']    
    tableAllFantasy.columns = columnsAllFantasy 
    
    SupercoachSoup = utlScrape.getSoup(hrefSupercoach)
    
    first_season = 1
    r = 0
    
    while r < 1:
        if first_season == 1:
            htmlSupercoach = SupercoachSoup.find("div", id=re.compile("player-fantasy-round-*"))
            first_season = 0
        elif first_season == 0:
            htmlSupercoach = htmlSupercoach.find_next("div", id=re.compile("player-fantasy-round-*"))
        
        try:
            htmlSupercoach['id']
        except TypeError:
            r = 1
            continue
            
        year = htmlSupercoach['id']
        year = int(year[21:])
        
        tableSupercoach  = htmlSupercoach.find('table',width = "688", border = "0")
        tableSupercoach = pd.read_html(str(tableSupercoach ))
        tableSupercoach = tableSupercoach[0]
        columnsSupercoach = tableSupercoach.iloc[0,:]
        tableSupercoach = pd.DataFrame(tableSupercoach.iloc[1:,:])
        tableSupercoach.columns = columnsSupercoach
        tableSupercoach['year'] = year
        
        try: 
            tableAllSupercoach 
        except NameError:
            tableAllSupercoach = tableSupercoach
        else: 
            tableAllSupercoach = pd.concat([tableAllSupercoach , tableSupercoach])
    
    columnsAllSupercoach = ['Round','SupercoachPrice','SupercoachScore', 'SupercoachValue','year']    
    tableAllSupercoach.columns = columnsAllSupercoach 
    
    tablePlayerFantasyScores = pd.merge(tableAllSupercoach,tableAllFantasy,on = ['Round','year'])
    tablePlayerFantasyScores['Player'] = PlayerLink.Player
    tablePlayerFantasyScores['fw_player_id'] = PlayerLink.fw_player_id
    del tableAllFantasy, tableAllSupercoach
    
    try: 
        dfAllFantasyScores 
    except NameError:
        dfAllFantasyScores = tablePlayerFantasyScores
    else: 
        dfAllFantasyScores = pd.concat([dfAllFantasyScores , tablePlayerFantasyScores])
    dfAllFantasyScores = dfAllFantasyScores.drop_duplicates()
    gc.collect()

    
dfAllFantasyScores['SupercoachPrice'] = utlScrape.cleanPrices(dfAllFantasyScores['SupercoachPrice'])
dfAllFantasyScores['AFLFantasyPrice'] = utlScrape.cleanPrices(dfAllFantasyScores['AFLFantasyPrice'])

dfAllFantasyScores = dfAllFantasyScores[['year','Round','SupercoachPrice','AFLFantasyPrice','fw_player_id']]
dfAllFantasyScores.columns = ['season_round','year','SupercoachPrice','AFLFantasyPrice','fw_player_id']
dfAllFantasyScores['season_round'] = dfAllFantasyScores['season_round'].astype('int')

if os.path.isfile('./data/dfAllFantasyScores.pkl'):
    dfAllFantasyScores_read = pd.read_pickle('data/dfAllFantasyScores.pkl')
    dfAllFantasyScores = pd.concat([dfAllFantasyScores_read, dfAllFantasyScores])
    dfAllFantasyScores = dfAllFantasyScores.drop_duplicates()
pd.to_pickle(dfAllFantasyScores, 'data/dfAllFantasyScores.pkl')