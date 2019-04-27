import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

afl_ML = pd.read_pickle('afl_ML.pkl')
#Only want to test on newer games
test_candidates = afl_ML[afl_ML.year > 2013][['fw_game_id','round_index']]
fw_game_ids = pd.DataFrame(test_candidates
            ).sort_values(by = ['fw_game_id']
                         ).drop_duplicates(subset = 'fw_game_id'
                                          ).reset_index(drop = True)

np.random.seed(80085)
train_ids, test_ids, a, b = train_test_split(fw_game_ids, fw_game_ids.round_index, test_size=0.1)
del a, b, train_ids
test_ids.to_pickle("test_fw_ids.pkl")