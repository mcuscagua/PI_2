import numpy as np
import pandas as pd
from ta import *
import itertools as it

import warnings
warnings.filterwarnings("ignore")

def normalize(x,time):
    return (x-x.rolling(time, min_periods = 1).min())/(x.rolling(time, min_periods = 1).max()-x.rolling(time, min_periods = 1).min())

def max_rate(x,time):
    return x/x.rolling(time, min_periods = 1).max()

def min_rate(x,time):
    return x/x.rolling(time, min_periods = 1).min()

def past_return(x,n):
    y = pd.DataFrame()
    for i in range(n):
        key = 'T-'+str(i+1)
        y[key] = x['Close']/x['Close'].shift(i+1)-1
    return y

class EigTrade():
    def __init__(self, DATA, FEAT, dr, Time_Window, P_Ret, max_min_tw, n_means, windows_ref, precision, long_short):
        self.dr = dr                        # Desired return
        self.Time_Window = Time_Window      # Time window to see into the future
        self.P_Ret = P_Ret                  # Past return to consider
        self.max_min_tw = max_min_tw        # timewindow for support and resistance
        self.n_means = n_means              # moving averages for features
        self.windows_ref = windows_ref      # past information needed to normalize
        self.precision = precision          # precision to pick the number of eigen vectors to use
        self.long_short = long_short        # long-recon, long-space, short-space, short-space
        self.hist_data = DATA
        self.hist_feature = FEAT

        self.Dates_of_Interest = pd.DataFrame(columns=self.hist_data[[x for x in self.hist_data.keys()][0]].columns)

        self.epsilon_distance = 24.379236091501802
        self.epsilon_reconstruction = 48.960341149199174

        self.dropable_col = ['volatility_kchi', 'volatility_kcli', 'trend_adx', 'trend_adx_pos',
                             'volatility_atr', 'volatility_bbhi', 'volatility_dchi', 'trend_adx_neg']

    def getHistFeat(self):
        self.HF = pd.DataFrame(columns=self.hist_feature.keys())

        for feat, df in self.hist_feature.items():
            df_aux = df.copy()
            self.HF[feat] = df_aux['Close']
            for i in it.product(*self.n_means):
                self.HF[feat+ '_MAS_' + str(i[1]) + "_" + str(i[0])] = df_aux['Close'].rolling(window=i[1],
                                                                                           min_periods=1).mean() - \
                                                                   df_aux['Close'].rolling(window=i[0],
                                                                                           min_periods=1).mean()
                self.HF[feat + '_MAR_' + str(i[1]) + "_" + str(i[0])] = df_aux['Close'].rolling(window=i[1],
                                                                                           min_periods=1).mean() / \
                                                                   df_aux['Close'].rolling(window=i[0],
                                                                                           min_periods=1).mean()
                self.HF[feat + '_EMAS_' + str(i[1]) + "_" + str(i[0])] = df_aux['Close'].ewm(span=i[1], adjust=False,
                                                                                        min_periods=1).mean() - df_aux[
                                                                        'Close'].ewm(span=i[0], adjust=False,
                                                                                     min_periods=1).mean()
                self.HF[feat + '_EMAR_' + str(i[1]) + "_" + str(i[0])] = df_aux['Close'].ewm(span=i[1], adjust=False,
                                                                                        min_periods=1).mean() / df_aux[
                                                                        'Close'].ewm(span=i[0], adjust=False,
                                                                                     min_periods=1).mean()
        self.HF = self.HF.apply(normalize, time=self.windows_ref)

    def getDatesInterest(self):
        self.dates_indexes = {}
        for key, value in self.hist_data.items():
            if 'long' in self.long_short:
                self.dates_indexes[key] = value['Close'] / value['Close'].shift(-self.Time_Window) - 1 > self.dr
            else:
                self.dates_indexes[key] = value['Close'] / value['Close'].shift(-self.Time_Window) - 1 < self.dr



    def buildHistData(self):
        self.getDatesInterest()
        for share, df1 in self.hist_data.items():
            df_aux = df1.copy()
            df_aux = add_all_ta_features(df_aux, "Open", "High", "Low", "Close", "Volume", fillna=True)
            df_aux['Vol_Max'] = max_rate(df_aux['Volume'], self.max_min_tw)
            df_aux['Clo_Max'] = max_rate(df_aux['Close'], self.max_min_tw)
            df_aux['Ope_Max'] = max_rate(df_aux['Open'], self.max_min_tw)
            df_aux['Hig_Max'] = max_rate(df_aux['High'], self.max_min_tw)
            df_aux['Low_Max'] = max_rate(df_aux['Low'], self.max_min_tw)

            df_aux['Vol_Min'] = min_rate(df_aux['Volume'], self.max_min_tw)
            df_aux['Clo_Min'] = min_rate(df_aux['Close'], self.max_min_tw)
            df_aux['Ope_Min'] = min_rate(df_aux['Open'], self.max_min_tw)
            df_aux['Hig_Min'] = min_rate(df_aux['High'], self.max_min_tw)
            df_aux['Low_Min'] = min_rate(df_aux['Low'], self.max_min_tw)

            df_aux['Support'] = df_aux['Close']/df_aux['Low'].rolling(window=self.max_min_tw, min_periods=1).median()-1
            df_aux['Resistance'] = df_aux['Close']/df_aux['High'].rolling(window=self.max_min_tw,min_periods=1).median()-1

            df_aux = df_aux.apply(normalize, time=self.windows_ref)
            df_aux = df_aux.drop(self.dropable_col, axis=1)

            df_past_r = past_return(df1, n=self.P_Ret)
            df_aux = pd.concat([df_aux, df_past_r, self.HF], axis=1)

            self.hist_data[share] = df_aux

        for share, df in self.hist_data.items():
            self.Dates_of_Interest = self.Dates_of_Interest.append(df.iloc[self.dates_indexes[share].values])

        self.Dates_of_Interest = self.Dates_of_Interest.dropna().transpose()

    def BuildTradeSpace(self):
        self.Phi_Mat_regular = np.array(self.Dates_of_Interest.values, dtype=float)
        self.phi_mean = np.reshape(np.mean(self.Phi_Mat_regular, axis=1), (-1, 1))
        self.Phi_Mat_centered = self.Phi_Mat_regular - self.phi_mean

        self.CM = np.dot(self.Phi_Mat_centered, self.Phi_Mat_centered.transpose())
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.CM)
        self.eig_pairs = [(self.eigenvalues[index], self.eigenvectors[:, index]) for index in range(len(self.eigenvalues))]

        self.eig_pairs.sort(reverse=True)
        self.eigvalues_sort = [self.eig_pairs[index][0] for index in range(len(self.eigenvalues))]
        self.eigvectors_sort = [self.eig_pairs[index][1] for index in range(len(self.eigenvalues))]

        self.K = sum(np.cumsum(self.eigvalues_sort) / sum(self.eigvalues_sort) < self.precision)

        self.reduced_eig = np.array(self.eigvectors_sort[:self.K])
        self.eig_trade = np.dot(self.reduced_eig, self.Phi_Mat_centered)

    def main(self):
        self.getHistFeat()
        self.buildHistData()
        self.BuildTradeSpace()

    def getDistance(self, day, mode = 'long-recon'):

        ob_cent = day.values - self.phi_mean[:, 0]
        weights = np.dot(self.reduced_eig, ob_cent).reshape((1, self.K))
        ghost_trade = np.dot(weights, self.reduced_eig).transpose() + self.phi_mean
        Recon = np.linalg.norm(day.values - ghost_trade)
        dist_list = [np.linalg.norm(self.eig_trade[:, j] - weights.transpose()) for j in range(self.eig_trade.shape[1])]
        minESpace = min(dist_list)

        if mode == 'long-recon' and Recon < self.epsilon_reconstruction:
            return 1
        else:
            return 0

        if mode != 'long-space' and minESpace < self.epsilon_distance:
            return 1
        else:
            return 0

        if mode == 'short-recon' and Recon < self.epsilon_reconstruction:
            return -1
        else:
            return 0

        if mode != 'short-space' and minESpace < self.epsilon_distance:
            return -1
        else:
            return 0