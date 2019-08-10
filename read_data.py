# -*- coding: utf-8 -*-
"""
@author: jcuscagu
"""
import pandas as pd
import datetime as dt

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