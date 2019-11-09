setwd("C:\\Users\\mcusc\\Documents\\Maestria\\S2\\PI_2\\Datos")
archivos1 = dir()

setwd("D:\\BD Mauro")
archivos2 = dir()

archivos1 = unique(sapply(archivos1, function(x){strsplit(x,"_")[[1]][1]}))
archivos2 = unique(sapply(archivos2, function(x){strsplit(x,"_")[[1]][1]}))
