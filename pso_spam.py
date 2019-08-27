from pso import *

res_xgboost = []
for i in range(0, 1000):
	pso = PSO('../data/spambase.csv', 'class', 'xgboost')
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("spam_xgboost_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_xgboost[i]+"\n")
f_pok.close()

res_svm = []
for i in range(0, 1000):
	pso = PSO('../data/spambase.csv', 'class', 'svm')
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("spam_svm_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_svm[i]+"\n")
f_pok.close()

res_tree = []
for i in range(0, 1000):
	pso = PSO('../data/spambase.csv', 'class', 'tree')
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("spam_tree_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_tree[i]+"\n")
f_pok.close()