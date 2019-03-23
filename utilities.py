import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def newMeanCol(df, col_nm, N):
    new_col = col_nm + '_mean' + str(N)
    x = df.loc[0:,col_nm]
    mean_vals = list(np.insert(
        running_mean(
            np.array(x), N), 0, [np.nan] * N))
    df[new_col] = mean_vals[0:len(x)]
    return df

def colLookup(data, string):
    return data[[col for col in data.columns if string  in col]]

#Where a row is na (i.e. it is in the first x games for that team and therefore has no mean), remove both rows of that game.
#This is for the instance when new teams join the league, can't calculate the score for the older team so have to remove both instances.
def NANConsolidate(df, col):
    na_game_ids = set(df[(df.isna().any(axis = 1))][col])
    return df[~df[col].isin(na_game_ids)]

def CheckTeam(df, team):
    df = df[df['team'] == team]
    min_game = min(df.date)
    max_game = max(df.date)
    n_na = sum(df.isna().any(axis=1))
    return print(min_game, max_game, n_na)
