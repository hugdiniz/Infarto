matrix = readcsv("Banco de Dados - Infarto treinoTeste.csv")
indexInfarto = find(x->x==1,matrix[:,4])

sizeRange = 12

values = sort(unique(matrix[indexInfarto,1]))
valueRange = values[1]:round(Int,abs(values[1] - values[end])/sizeRange):values[end]
colesterolInfartou = [count(i->x<=i<x+sizeRange,matrix[indexInfarto,1]) for x in valueRange]
colesterolTotal = [count(i->x<=i<x+sizeRange,matrix[2:end,1]) for x in valueRange]
nameRange = [string(x," ate ",x+round(Int,abs(values[1] - values[end])/sizeRange)-1) for x in valueRange]
colesterolResultado = colesterolInfartou ./ colesterolTotal

values = sort(unique(matrix[indexInfarto,2]))
valueRange = values[1]:round(Int,abs(values[1] - values[end])/sizeRange):values[end]
idadeInfartou = [count(i->x<=i<x+sizeRange,matrix[indexInfarto,2]) for x in valueRange]
idadeTotal = [count(i->x<=i<x+sizeRange,matrix[2:end,2]) for x in valueRange]
nameRange = [string(x," ate ",x+round(Int,abs(values[1] - values[end])/sizeRange)-1) for x in valueRange]
idadeResultado = idadeInfartou ./ idadeTotal

values = sort(unique(matrix[indexInfarto,3]))
valueRange = values[1]:round(Int,abs(values[1] - values[end])/sizeRange):values[end]
glicemiaInfartou = [count(i->x<=i<x+sizeRange,matrix[indexInfarto,3]) for x in valueRange]
glicemiaTotal = [count(i->x<=i<x+sizeRange,matrix[2:end,3]) for x in valueRange]
nameRange = [string(x," ate ",x+round(Int,abs(values[1] - values[end])/sizeRange)-1) for x in valueRange]
glicemiaResultado = glicemiaInfartou ./ glicemiaTotal



saida = hcat(colesterolResultado,idadeResultado,glicemiaResultado)
writecsv("Banco de Dados - Infarto teste formatado.csv",saida)