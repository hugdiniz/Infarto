function splitKFold(y, num_folds)
  i = shuffle([1:y]);
  fold_size = int(floor(y/num_folds));
  remainder = y-num_folds*fold_size;
  groups = zeros(Int, y);
  cursor = 1;
  group = 1;

  while cursor<=y
    this_fold_size = group <= remainder ? fold_size+1:fold_size;
    groups[i[cursor:cursor+this_fold_size-1]] = group;
    group += 1;
    cursor += this_fold_size;
  end

  return groups;
end

folds = 10
csv = readcsv("Banco de Dados - Infarto treinoTeste.csv")
titulo = csv[1,:]
csv = csv[2:end,:]
index = splitKFold(size(csv)[1], folds)

treinoTesteIndex = find(r -> r != 1, index)
validacaoIndex = find(r -> r == 1, index)

csvValidacao = vcat(titulo,csv[validacaoIndex,:])[:,1:4]
csvTreinoTeste = vcat(titulo,csv[treinoTesteIndex,:])[:,1:4]

writecsv("Banco de Dados - Infarto validacao.csv",csvValidacao)
writecsv("Banco de Dados - Infarto treinoTeste.csv",csvTreinoTeste)
