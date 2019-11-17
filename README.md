# PI_2

Trabajo desarrollado por:

- Liceth Mosquera
- Juan Diego Estrada
- Javier Rosso
- Johan Rios
- Juan Mauricio Cuscagua

### Datos:

- En la carpeta Datos se encuentran los csv para construir las series de tiempo de los precios de los activos usados en el ejercicio.

### Generador de Backtesting de estrategias iniciales:

- Los notebooks "Backtesting_Estrategias", "Backtesting_Estrategias_Individuales", "Backtesting_Estrategias_Regionales" y "Backtesting_Estrategias_Sectoriales" contienen el backtesting de las estrategias de factores usando los diferentes conjuntos de activos.

### Backtesting de clustering de estrategias:
Encontrará 3 archivos:

- 1_Backtesting_General_Clasificadores_Sectoriales: Backtesting de los algoritmos de clustering sobre las estrategias de la canasta de ETF
- 2_Backtesting_General_Clasificadores_Individuales: Backtesting de los algoritmos de clustering sobre las estrategias de la canasta de activos individuales
- 3_Backtesting_General_Clasificadores_Conjunta: Backtesting de los algoritmos de clustering sobre las estrategias de la canasta formada con todos los activos disponibles

### Archivos de apoyo:

- Alphas101: módulo de python construido para facilitar la invocación de las funciones que calculan los factores Alpha.
- read_data: módulo de python construido para la lectura de los datos.

### Archivos csv:
 - Aquellos cuyo título inicia con "Rentabilidad de estrategias" corresponden a los retornos diarios de las estrategias a revisar para formar clusters.
 - Aquellos cuyo título inicia con "Cluster Rent" representan las rentabilidades diarias del backtesting de los algoritmos de clustering.
 
 
Finalmente, el notebook Analisis de Resultados genera las gráficas implementadas en la presentación así como la tabla de resultados de los algoritmos.
