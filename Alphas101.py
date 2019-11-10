import numpy as np


def alpha1(df, parameters = [20, 5]):
    """
    df: Un Dataframed
    parameters: Una tupla con los valores de los parametros
    """
    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)
    df_aux['stddev'] = df_aux['ret'].rolling(parameters[0]).std()
    df_aux['returns < 0'] = np.where(df_aux['ret'] < 0, df_aux['stddev'], df_aux['Close'])
    df_aux['(returns < 0)^2'] = df_aux['returns < 0'] ** 2
    df_aux['Ts_ArgMax'] = df_aux['(returns < 0)^2'].rolling(parameters[1]).apply(np.argmax)
    factor = df_aux['Ts_ArgMax'].rank()
    return factor[-1]


def alpha2(df, parameters = [1,6]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()
    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    df_aux['log(volume)'] = np.log(df_aux['Volume'])
    df_aux['delta(log(volume), 2)'] = df_aux['log(volume)'] - df_aux['log(volume)'].shift(parameters[0])
    df_aux['rank_volume'] = df_aux['delta(log(volume), 2)'].rank()
    df_aux['((close - open) / open)'] = (df_aux['Close'] - df_aux['Open']) / df_aux['Open']
    df_aux['rank_intradia'] = df_aux['((close - open) / open)'].rank()
    df_aux['corr'] = df_aux['rank_intradia'].rolling(parameters[1]).corr(df_aux['rank_volume'])

    factor = df_aux['corr'] * -1

    return factor[-1]



def alpha3(df, parameters = [60,10]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    df_aux['zopen'] = (df_aux['Open'] - df_aux['Open'].rolling(parameters[0]).mean())/df_aux['Open'].rolling(parameters[0]).std()
    df_aux['zvolume'] = (df_aux['Volume'] - df_aux['Volume'].rolling(parameters[0]).mean())/df_aux['Volume'].rolling(parameters[0]).std()
    df_aux['rank(open)'] = df_aux['zopen'].rank()
    df_aux['rank(volume)'] = df_aux['zvolume'].rank()
    df_aux['corr'] = df_aux['rank(open)'].rolling(parameters[1]).corr(df_aux['rank(volume)'])

    factor = df_aux['corr'] * -1

    return factor[-1]


def alpha6(df, parameters = [10]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)
    df_aux['corr'] = df_aux['Open'].rolling(parameters[0]).corr(df_aux['Volume'])

    factor = df_aux['corr'] * -1

    return factor[-1]


def alpha44(df, parameters = [5]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    df_aux['rank(volume)'] = df_aux['Volume'].rank()
    df_aux['corr'] = df_aux['High'].rolling(parameters[0]).corr(df_aux['rank(volume)'])

    factor = df_aux['corr'] * -1

    return factor[-1]


def alpha53(df, parameters = [9]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    df_aux['cambios'] = ((df_aux['Close'] - df_aux['Low']) - (df_aux['High'] - df_aux['Close']))
    df_aux['cambios'] = df_aux['cambios'] / (df_aux['Close'] - df_aux['Low'] + 0.0001)

    df_aux['delta(cambios)'] = df_aux['cambios'] - df_aux['cambios'].shift(parameters[0])

    factor = df_aux['delta(cambios)'] * -1

    return factor[-1]


def alpha54(df):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    factor = (-1 * (df_aux['Low'] - df_aux['Close'] + 0.0001) * (df_aux['Open'] ** 5))
    factor = factor / ((df_aux['Low'] - df_aux['High'] + 0.0001) * (df_aux['Open'] ** 5))

    return factor[-1]


def alpha101(df, parameters = [1]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    factor = (df_aux['Close'] - df_aux['Open']) / ((df_aux['High'] - df_aux['Low']) + 0.001)

    return factor[-1]


def mar(df, parameters = [20,200]):
    """
    df: Un Dataframe
    parameters: Una tupla con los valores de los parametros
    """

    df_aux = df.copy()

    # Siempre debe estar la siguiente linea
    df_aux['ret'] = df_aux['Close'].pct_change()  # returns
    df_aux['ret'] = df_aux['ret'].fillna(0)

    df_aux['FMA'] = df_aux['Close'].rolling(parameters[0]).mean()
    df_aux['SMA'] = df_aux['Close'].rolling(parameters[1]).mean()

    factor = df_aux['FMA'] / df_aux['SMA']

    return factor[-1]