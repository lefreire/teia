from pso import *

res_xgboost = []
for i in range(0, 10):
	pso = PSO('../data/pokemon.csv', 'is_legendary', 'xgboost')
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("pok_xgboost_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_xgboost[i]+"\n")
f_pok.close()

res_svm = []
for i in range(0, 1000):
	pso = PSO('../data/pokemon.csv', 'is_legendary', 'svm')
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("pok_svm_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_svm[i]+"\n")
f_pok.close()

res_tree = []
for i in range(0, 1000):
	pso = PSO('../data/pokemon.csv', 'is_legendary', 'tree')
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("pok_tree_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_tree[i]+"\n")
f_pok.close()

res_xgboost = []
pso = PSO('../data/pd_speech_features.csv', 'class', 'xgboost')
for i in range(0, 1000):
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("pd_xgboost_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_xgboost[i]+"\n")
f_pok.close()

res_svm = []
pso = PSO('../data/pd_speech_features.csv', 'class', 'svm')
for i in range(0, 1000):
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("pd_svm_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_svm[i]+"\n")
f_pok.close()

res_tree = []
pso = PSO('../data/pd_speech_features.csv', 'class', 'tree')
for i in range(0, 1000):
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("pd_tree_pso.txt","w+")
for i in range(0, 1000):
     f_pok.write(res_tree[i]+"\n")
f_pok.close()


