# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:21:45 2017

@author: fhgomez
"""
import pandas as pd
import datetime as dt
import os
import pdb

import sys
sys._enablelegacywindowsfsencoding() # esto es para cambiar el encoding y que pueda buscar en la carpeta del G, ya que hay una tilde ahi que

directory = 'Datos'

def reemplazar(string,old,new):
    # Reemplazo los signos # con nada
    str1 = string
    str1 = str1.replace(old,new)
    return str1

class anuales:
    def importar(self,file, sep = ',', delimiter = None):
        # Importo los datos desde una direccion
        self.df = pd.read_csv(file, sep = sep, delimiter = delimiter)
    def quitar_num(self, columna,file, sep = ',', delimiter = None):
        # Le quito los signos # a todo una serie de un DF
        self.importar(file,sep,delimiter)
        dates = self.df[columna]
        dates = dates.apply(reemplazar,args = ('#',''))
        self.df['Date'] = dates
    def indice_fechas(self,columna,file, sep = ',', delimiter = None):
        # Convierto los indices en fechas y borro la columna Date
        self.quitar_num(columna,file, sep = ',', delimiter = None)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format = '%Y-%m-%d')
        self.df.index = self.df['Date']; del self.df['Date']

def crearDF(simbolo,años, Features = False):
    DF = pd.DataFrame()
    for i in años:
        activo = anuales()
        if Features:
            activo.indice_fechas('Date',directory+'\\'+simbolo+'_1 Dia_'+str(i)+'.csv')
        else:
            activo.indice_fechas('Date',directory+'\\'+simbolo+' US Equity_1 Dia_'+str(i)+'.csv')
        
        activo.df = activo.df.iloc[~activo.df.index.duplicated()]
        DF = DF.append(activo.df)
    return DF

def organizarTodo(simbolos, años, Features = False):
    """
    simbolos: es una lista que contiene los simbolos que se desean importar
    años: una lista con los años que se desean utilizar
    """
    diccionario = {}
    for j in simbolos:
        diccionario[j] = crearDF(j,años,Features)
        diccionario[j] = diccionario[j].loc[~diccionario[j].index.duplicated()]
    return diccionario


import itertools
import datetime as t
import numpy as np


def raw_pipeline(simbolos,años, Features = False):
    bd_semi = organizarTodo(simbolos, años, Features)
    
    last_days_str = []
    activos_index = []
    for i in bd_semi[list(bd_semi.keys())[0]].index:
        temp = t.datetime.strftime(i,'%Y-%m-%d')
        last_days_str.extend(np.repeat(temp,len(simbolos)))
        activos_index.extend(simbolos)
    
    opens = [] # lista con los closes
    highs = [] # lista con los closes
    lows = [] # lista con los closes
    closes = [] # lista con los closes
    volumes = [] # lista con los closes
    
    for key,value in bd_semi.items():
        opens.append(list(value['Open']))
        highs.append(list(value['High']))
        lows.append(list(value['Low']))
        closes.append(list(value['Close']))
        volumes.append(list(value['Volume']))
    
    l_opens = list(itertools.chain(*zip(*opens)))
    l_highs = list(itertools.chain(*zip(*highs)))
    l_lows = list(itertools.chain(*zip(*lows)))
    l_closes = list(itertools.chain(*zip(*closes)))
    l_volumes = list(itertools.chain(*zip(*volumes)))
    
    indices = [last_days_str,activos_index]
    tuplas = list(zip(*indices))
    multi_indice = pd.MultiIndex.from_tuples(tuplas, names=['date', 'symbol'])
    
    bd = pd.DataFrame(pd.DataFrame([l_opens,l_highs,l_lows,l_closes,l_volumes]).T.values, index = multi_indice)
    bd.columns = ['Open','High','Low','Close','Volume']
    
    return bd

def open2(bd_semi,simbolos):

    for sim in simbolos:
        directo = 'G:\Centralización de Inventarios\ACCIONES\Trading Algoritmico\BD intradia2\\'+sim+'.csv'
        
        a = pd.read_csv(directo)
        a['Date'] = pd.to_datetime(a['Date'],format = "%Y-%m-%d %H:%M:%S")
        a.index = a['Date'];del a['Date']
        
        af = a.loc[(a.index.hour == 10)&(a.index.minute == 0)]
        
        delta = 10*60*60
        
        af.index = af.index + dt.timedelta(0,-delta)

        aapl = bd_semi[sim]

        af2 = af.loc[np.in1d(af.index,aapl.index),"Open"]
        af2 = af2.loc[~af2.index.duplicated()]

        aapl['Open2'] = af2
#        aapl = aapl.dropna()
        aapl['Open2'] = aapl['Open2'].fillna(method = "ffill")

        bd_semi[sim] = aapl.copy()

    return bd_semi

def raw_pipeline_op2(simbolos,años):
    bd_semi = organizarTodo(simbolos, años)
    bd_semi = open2(bd_semi,simbolos)
    
#    return bd_semi
    
    last_days_str = []
    activos_index = []
    for i in bd_semi[list(bd_semi.keys())[0]].index:
        temp = t.datetime.strftime(i,'%Y-%m-%d')
        last_days_str.extend(np.repeat(temp,len(simbolos)))
        activos_index.extend(simbolos)
    
    opens = [] # lista con los closes
    highs = [] # lista con los closes
    lows = [] # lista con los closes
    closes = [] # lista con los closes
    volumes = [] # lista con los closes
    opens2 = [] 
    
    for key,value in bd_semi.items():
#        print(key)
        opens.append(list(value['Open']))
        highs.append(list(value['High']))
        lows.append(list(value['Low']))
        closes.append(list(value['Close']))
        volumes.append(list(value['Volume']))
        opens2.append(list(value['Open2']))
    
    l_opens = list(itertools.chain(*zip(*opens)))
    l_highs = list(itertools.chain(*zip(*highs)))
    l_lows = list(itertools.chain(*zip(*lows)))
    l_closes = list(itertools.chain(*zip(*closes)))
    l_volumes = list(itertools.chain(*zip(*volumes)))
    l_opens2 = list(itertools.chain(*zip(*opens2)))
    
    indices = [last_days_str,activos_index]
    tuplas = list(zip(*indices))
    multi_indice = pd.MultiIndex.from_tuples(tuplas, names=['date', 'symbol'])
    
    bd = pd.DataFrame(pd.DataFrame([l_opens,l_highs,l_lows,l_closes,l_volumes,l_opens2]).T.values, index = multi_indice)
    bd.columns = ['Open','High','Low','Close','Volume','Open2']
    
    bd['Open2'] = np.where(np.isnan(bd['Open']),np.nan,bd['Open2'])
    
    return bd


def imp_kibot(años,hora,minuto):
#    años = [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
#    hora = 10
#    minuto = 0
    
    year = str(años[0])
    #limite = '2018-11-26'
    
    direccion = 'D:\Compartida\Python 3\BD Fabio\Daily\\'
    files = os.listdir(direccion) # returns list
    
    bd_dicc = {}
    for i in files:
        name = i.replace('.txt','')
        df = pd.read_table(direccion+i, sep =',', header = None)
        df.columns = ['Date','Open','High','Low','Close','Volume']
        df['Date'] = pd.to_datetime(df['Date'],format = '%m/%d/%Y'); df.index = df['Date']; del df['Date']
    #    df2 = df.loc[df.index <= limite]
        bd_dicc[name] = df.loc[df.index >= year].copy()
    
    
    direccion = 'D:\Compartida\Python 3\BD Fabio\\30Min\\'
    files = os.listdir(direccion) # returns list
    
    bd_dicc_30 = {}
    sym = []
    for i in files:
        name = i.replace('.txt','')
        sym.append(name)
        df = pd.read_table(direccion+i, sep =',', header = None)
        df.columns = ['Date','Hour','Open','High','Low','Close','Volume']
        df['DateH'] = df['Date']+' '+df['Hour']; del df['Date']; del df['Hour']
        
        df['DateH'] = pd.to_datetime(df['DateH'],format = '%m/%d/%Y %H:%M'); df.index = df['DateH']; del df['DateH']
        df2 = df.loc[(df.index.hour == hora) & (df.index.minute == minuto)]
        bd_dicc_30[name] = df2.loc[df2.index >= year].copy()
        bd_dicc[name]['Open2'] = bd_dicc_30[name]['Open'].values
    
    return bd_dicc, sym

def raw_pipeline2(años):
    bd_semi,simbolos = imp_kibot(años, 10, 0)
    
    last_days_str = []
    activos_index = []
    for i in bd_semi[list(bd_semi.keys())[0]].index:
        temp = t.datetime.strftime(i,'%Y-%m-%d')
        last_days_str.extend(np.repeat(temp,len(simbolos)))
        activos_index.extend(simbolos)
    
    opens = [] # lista con los closes
    highs = [] # lista con los closes
    lows = [] # lista con los closes
    closes = [] # lista con los closes
    volumes = [] # lista con los closes
    opens2 = []
    
    for key,value in bd_semi.items():
        opens.append(list(value['Open']))
        highs.append(list(value['High']))
        lows.append(list(value['Low']))
        closes.append(list(value['Close']))
        volumes.append(list(value['Volume']))
        opens2.append(list(value['Open2']))
    
    l_opens = list(itertools.chain(*zip(*opens)))
    l_highs = list(itertools.chain(*zip(*highs)))
    l_lows = list(itertools.chain(*zip(*lows)))
    l_closes = list(itertools.chain(*zip(*closes)))
    l_volumes = list(itertools.chain(*zip(*volumes)))
    l_opens2 = list(itertools.chain(*zip(*opens2)))
    
    indices = [last_days_str,activos_index]
    tuplas = list(zip(*indices))
    multi_indice = pd.MultiIndex.from_tuples(tuplas, names=['date', 'symbol'])
    
    #    pdb.set_trace()
    
    bd = pd.DataFrame(pd.DataFrame([l_opens,l_highs,l_lows,l_closes,l_volumes,l_opens2]).T.values, index = multi_indice)
    bd.columns = ['Open','High','Low','Close','Volume','Open2']
    
    return bd







































