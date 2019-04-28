year_start = 2007
year_end = 2019

import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import re
import os
import sys

import utilities.utilitiesScrape as utlScrape

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
    
if os.path.isfile('./afl_DF.pkl'):
    afl_DF = pd.read_pickle('afl_DF.pkl')

if 'afl_DF' in locals():
    scraped_ids = list(afl_DF.loc[0:]['fw_game_id'].astype('int'))
    game_ids = [x for x in game_ids if x not in scraped_ids]
    
for game_id in game_ids:
    #scrape the data from the game page on footywire.com
    game_id =  str(game_id)
    afl_url = 'https://www.footywire.com/afl/footy/ft_match_statistics?mid=' + game_id
    try:
        soup = utlScrape.getSoup(afl_url)
    except:
        continue
    new_game = utlScrape.scrapeGameData(soup, game_id = game_id, afl_url = afl_url) 
    if 'afl_update' not in locals():
        afl_update = new_game
    else:
        afl_update  = afl_update.append(new_game, ignore_index=True)
    sys.stdout.flush()
    print(min(new_game.year),
    min(new_game.round_char))
    
if 'afl_update' in locals():
    if 'afl_DF' not in locals():
        afl_DF = afl_update
    else:
        afl_DF = afl_DF.append(afl_update, ignore_index = True, sort = True)
    afl_DF.to_pickle('afl_DF.pkl')    
    del afl_update
else:
    print('afl_update does not exist')
