from pre_process import PreProcessing
from models import XGBoostModel, SVMModel, DecisionTreeModel

pre_processing = PreProcessing('../data/pokemon.csv')
data = pre_processing.load_dataset()
x, y = pre_processing.separate_data(data, 'is_legendary')

print("COMECANDO O POKEMON")
res_xgboost = []
print("COMECANDO XGBOOST")
for i in range(0, 1000):
	xgbmodel = XGBoostModel(x, y)
	res_xgboost.append(xgbmodel.kfold_accuracy())
f_pok = open("pok_xgboost_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
print("COMECANDO SVM")
for i in range(0, 1000):
	svmmodel = SVMModel(x, y)
	res_svm.append(svmmodel.kfold_accuracy())
f_pok = open("pok_svm_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
print("COMECANDO DECISION TREE")
for i in range(0, 1000):
	treemodel = DecisionTreeModel(x, y)
	res_tree.append(treemodel.kfold_accuracy())
f_pok = open("pok_tree_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()

pre_processing = PreProcessing('../data/pd_speech_features.csv')
data = pre_processing.load_dataset()
x, y = pre_processing.separate_data(data, 'is_class')

print("COMECANDO O PD")
res_xgboost = []
print("COMECANDO XGBOOST")
for i in range(0, 1000):
	xgbmodel = XGBoostModel(x, y)
	res_xgboost.append(xgbmodel.kfold_accuracy())
f_pok = open("pd_xgboost_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
print("COMECANDO SVM")
for i in range(0, 1000):
	svmmodel = SVMModel(x, y)
	res_svm.append(svmmodel.kfold_accuracy())
f_pok = open("pd_svm_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
print("COMECANDO DECISION TREE")
for i in range(0, 1000):
	treemodel = DecisionTreeModel(x, y)
	res_tree.append(treemodel.kfold_accuracy())
f_pok = open("pd_tree_def.txt","w+")
for i in range(0, 1000):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()

