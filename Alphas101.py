import pandas as pd
import numpy as np


def to_rt(serie):
    return serie / serie.shift(1) - 1


def stddev(serie):
    return serie.rolling(20).std()


def ts_argmax(serie):
    return serie.rolling(5).apply(np.argmax)


def to_log(serie):
    return serie.apply(np.log)


def to_delta(serie, w):
    #         pdb.set_trace()
    delta = serie - serie.shift(w)
    return delta


def to_corr(d, col1, col2, dfm):
    try3 = pd.DataFrame()
    for i in list(dfm.index.get_level_values(1).unique()):
        try2 = dfm[dfm.index.get_level_values(1) == i]
        try2['corr'] = try2[col1].rolling(d).corr(try2[col2])
        try3 = pd.concat([try3, try2], axis=0)
    try3 = try3.sort_index(level=0)
    return try3


class Alphas():
    def __init__(self):
        self.alphas = 'Included alpahas: 1, 2, 3, 6, 44, 53, 54, 101'                        # Desired return

    def alpha1(self, df, parameters = [20, 5]):
        """
        df: Un Dataframe
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


    def alpha2(self, df, parameters = [1,6]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        df_aux = df.copy()
        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux['Close'].pct_change()  # returns
        df_aux['ret'] = df_aux['ret'].fillna(0)

        df_aux['log(volume)'] = np.log(df_aux['Volume'])
        df_aux['delta(log(volume), 2)'] = to_delta(df_aux['log(volume)'],parameters[0])
        df_aux['rank_volume'] = df_aux['delta(log(volume), 2)'].rank()
        df_aux['((close - open) / open)'] = (df_aux['Close'] - df_aux['Open']) / df_aux['Open']
        df_aux['rank_intradia'] = df_aux['((close - open) / open)'].rank()
        df_aux = to_corr(parameters[1], 'rank_intradia', 'rank_volume', df_aux)

        df_aux['factor'] = df_aux['corr'] * -1

        return df_aux


    def alpha3(df, parameters = [60,10]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        def to_z(serie, w):
            #         pdb.set_trace()
            return (serie - serie.rolling(w).mean()) / serie.rolling(w).std()

        def to_corr(d, col1, col2, dfm):
            try3 = pd.DataFrame()
            for i in list(dfm.index.get_level_values(1).unique()):
                try2 = dfm[dfm.index.get_level_values(1) == i]
                try2['corr'] = try2[col1].rolling(d).corr(try2[col2])
                try3 = pd.concat([try3, try2], axis=0)
            try3 = try3.sort_index(level=0)
            return try3

        def rank(serie):
            return serie.rank()

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  # returns
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux['zopen'] = df_aux.groupby(grouper)['Open'].apply(to_z, parameters[0])  # open

        df_aux['zvolume'] = df_aux.groupby(grouper)['Volume'].apply(to_z, parameters[0])  # volume

        df_aux['rank(open)'] = df_aux.groupby(grouper2)['zopen'].apply(rank)  # rank(open)

        df_aux['rank(volume)'] = df_aux.groupby(grouper2)['zvolume'].apply(rank)  # rank(open)

        df_aux = to_corr(parameters[1], 'rank(open)', 'rank(volume)', df_aux)  # correlation(rank(open), rank(volume), 10))

        df_aux['factor'] = df_aux['corr'] * -1  # (-1 * correlation(rank(open), rank(volume), 10))

        return df_aux


    def alpha6(df, parameters = [10]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        def to_corr(d, col1, col2, dfm):
            try3 = pd.DataFrame()
            for i in list(dfm.index.get_level_values(1).unique()):
                try2 = dfm[dfm.index.get_level_values(1) == i]
                try2['corr'] = try2[col1].rolling(d).corr(try2[col2])
                try3 = pd.concat([try3, try2], axis=0)
            try3 = try3.sort_index(level=0)
            return try3

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  #
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux = to_corr(parameters[0], 'Open', 'Volume', df_aux)  # correlation(open,volume,10)

        df_aux['factor'] = df_aux['corr'] * -1  # (-1*correlation(open,volume,10))

        return df_aux


    def alpha44(df, parameters = [5]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        def to_corr(d, col1, col2, dfm):
            try3 = pd.DataFrame()
            for i in list(dfm.index.get_level_values(1).unique()):
                try2 = dfm[dfm.index.get_level_values(1) == i]
                try2['corr'] = try2[col1].rolling(d).corr(try2[col2])
                try3 = pd.concat([try3, try2], axis=0)
            try3 = try3.sort_index(level=0)
            return try3

        def rank(serie):
            return serie.rank()

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  # returns
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux['rank(volume)'] = df_aux.groupby(grouper2)['Volume'].apply(rank)  # rank(volume)

        df_aux = to_corr(parameters[0], 'High', 'rank(volume)', df_aux)  # correlation(high,rank(volume),5)

        df_aux['factor'] = df_aux['corr'] * -1  # (-1*correlation(high,rank(volume),5))

        return df_aux


    def alpha53(df, parameters = [9]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        def to_delta(serie, w):
            delta = serie - serie.shift(w)
            return delta

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  # returns
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux['cambios'] = ((df_aux['Close'] - df_aux['Low']) - (df_aux['High'] - df_aux['Close'])) / (
                    df_aux['Close'] - df_aux['Low'] + 0.0001)  # ((close-low)-(high-close))/(close-low)

        df_aux['delta(cambios)'] = df_aux.groupby(grouper)['cambios'].apply(to_delta, parameters[
            0])  # delta(((close-low)-(high-close))/(close-low),9)

        df_aux['factor'] = df_aux['delta(cambios)'] * -1  # (-1*delta(((close-low)-(high-close))/(close-low),9))

        return df_aux


    def alpha54(df, parameters = [1]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        def to_delta(serie, w):
            delta = serie - serie.shift(w)
            return delta

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  # returns
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux['factor'] = (-1 * (df_aux['Low'] - df_aux['Close'] + 0.0001) * (df_aux['Open'] ** 5)) / (
                    (df_aux['Low'] - df_aux['High'] + 0.0001) * (
                        df_aux['Open'] ** 5))  # ((-1*(low-close)*(open○^5))/((low-high)*(open^5)))

        df_aux['factor'] = df_aux['factor'] * (df_aux['ret'] + 1)  # modificacion para los paquetes

        return df_aux


    def alpha101(df, parameters = [1]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')
        grouper2 = df_aux.index.get_level_values('date')

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)  # returns
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        df_aux['factor'] = (df_aux['Close'] - df_aux['Open']) / ((df_aux['High'] - df_aux['Low']) + 0.001)

        return df_aux


    def mar(df, parameters = [[20],[200]]):
        """
        df: Un Dataframe
        parameters: Una tupla con los valores de los parametros
        """

        df_aux = df.copy()

        grouper = df_aux.index.get_level_values('symbol')

        def to_rt(serie):
            return serie / serie.shift(1) - 1

        # Siempre debe estar la siguiente linea
        df_aux['ret'] = df_aux.groupby(grouper)['Close'].apply(to_rt)
        df_aux['ret'] = df_aux['ret'].replace(np.nan, 0)

        def to_mean(serie, w):
            return serie.rolling(w).mean()

        df_aux['FMA'] = df_aux.groupby(grouper)['Close'].apply(to_mean, w=parameters[0])
        df_aux['SMA'] = df_aux.groupby(grouper)['Close'].apply(to_mean, w=parameters[1])

        df_aux['factor'] = df_aux['FMA'] / df_aux['SMA']  # distancia entre ambas ma

        return df_aux

