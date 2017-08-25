import sys
data = sys.stdin.readlines()
for datum in data:
  print(datum)
splitup = data[0].split(";")
num_vars = len(splitup)/2
for i in range(num_vars):
  varName = splitup[2*i+1][7:]
  varValLine = splitup[2*i]
  varVal = varValLine[varValLine.find("=")+1:]
  print(varVal)
