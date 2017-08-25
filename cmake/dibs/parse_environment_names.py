import sys
data_source = sys.argv
data = data_source#.readlines()
splitup = data[1].split(";")
num_vars = len(splitup)/2
list_out = []
for i in range(num_vars):
  varName = splitup[2*i+1][7:]
  varValLine = splitup[2*i]
  varVal = varValLine[varValLine.find("=")+1:]
  list_out.append(varName)
sys.stdout.write(";".join(list_out).strip("\r"))
