#pip install webdriver-manager
#pip install selenium 
alias = 'BrianTreg_T'
password = ''

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pickle

gameProbs = pickle.load(open("cmbAvgScore.pkl", 'rb')) 
colProbs = 'Win'

teamsMonash = {'Adelaide':'Adelaide',
        'Geelong':'Geelong',
        'Melbourne':'Melbourne',
        'Essendon':'Essendon',
        'Carlton':'Carlton',
        'Sydney':'Sydney',
        'G_W_Sydney':'GWS',
        'Richmond':'Richmond',
        'Brisbane':'Brisbane',
        'P_Adelaide': 'Port Adelaide',
        'Collingwood':'Collingwood',
        'W_Coast': 'West Coast',
        'W_Bulldogs':'Western Bulldogs',
        'Gold_Coast': 'Gold Coast',
        'Hawthorn':'Hawthorn',
        'Kangaroos': 'North Melbourne',
        'Fremantle':'Fremantle',
        'St_Kilda': 'St Kilda'}


driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('http://probabilistic-footy.monash.edu/~footy/tips.shtml')
inputAlias =  driver.find_element_by_name('name')
inputAlias.clear()
inputAlias.send_keys(alias)

inputPassword =  driver.find_element_by_name('passwd')
inputPassword.clear()
inputPassword.send_keys(password)

#inputRound =  driver.find_element_by_name('round')
#inputRound.send_keys('4')

inputPassword.send_keys(Keys.RETURN)

for i in range(9):
    gameId = str(i + 1)
    whichgame = 'whichgame' + gameId 
    game = 'game' + gameId
    gameInput =  driver.find_element_by_xpath("//input[@name='" + whichgame + "'][@value='home']")
    gameInput.click()

    gameHomeTeam =  gameInput.find_element_by_xpath("..").text
    gameHomeTeam = teamsMonash[gameHomeTeam]
    
    currGame = gameProbs[gameProbs['team'] == gameHomeTeam]
    currProb = currGame[colProbs]
    currProb = str(float(round(currProb,3)))
    
    inputHomeProb =  driver.find_element_by_name(game)
    inputHomeProb.clear()
    inputHomeProb.send_keys(currProb)
    
    
