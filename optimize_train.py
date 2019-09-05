from pso import *

res_xgboost = []
for i in range(0, 10):
	pso = PSO('../data/pokemon.csv', 'is_legendary')
	pso.define_type('xgboost')
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("pok_xgboost_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
for i in range(0, 10):
	pso = PSO('../data/pokemon.csv', 'is_legendary')
	pso.define_type('svm')
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("pok_svm_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
for i in range(0, 10):
	pso = PSO('../data/pokemon.csv', 'is_legendary')
	pso.define_type('tree')
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("pok_tree_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()

res_xgboost = []
for i in range(0, 10):
	pso = PSO('../data/pd_speech_features.csv', 'class')
	pso.define_type('xgboost')
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("i_pd_xgboost_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
for i in range(0, 10):
	pso = PSO('../data/pd_speech_features.csv', 'class')
	pso.define_type('svm')
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("i_pd_svm_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
for i in range(0, 10):
	pso = PSO('../data/pd_speech_features.csv', 'class')
	pso.define_type('tree')
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("i_pd_tree_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()


res_xgboost = []
for i in range(0, 10):
	pso = PSO('../data/spambase.csv', 'class')
	pso.define_type('xgboost')
	cost, pos = pso.optimize_pso()
	res_xgboost.append([cost, pos])

f_pok = open("spam_xgboost_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
for i in range(0, 10):
	pso = PSO('../data/spambase.csv', 'class')
	pso.define_type('svm')
	cost, pos = pso.optimize_pso()
	res_svm.append([cost, pos])

f_pok = open("spam_svm_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
for i in range(0, 10):
	pso = PSO('../data/spambase.csv', 'class')
	pso.define_type('tree')
	cost, pos = pso.optimize_pso()
	res_tree.append([cost, pos])

f_pok = open("spam_tree_pso.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()


