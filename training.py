from pre_process import PreProcessing
from models import XGBoostModel, SVMModel, DecisionTreeModel

pre_processing = PreProcessing('../data/pokemon.csv')
data = pre_processing.load_dataset()
X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'is_legendary')

print("COMECANDO O POKEMON")
res_xgboost = []
print("COMECANDO XGBOOST")
for i in range(0, 10):
	print(i)
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'is_legendary')
	xgbmodel = XGBoostModel(X_train, ytrain, X_test, ytest)
	res = xgbmodel.kfold_accuracy()
	# print(res)
	res_xgboost.append(res)
f_pok = open("pok_xgboost_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
print("COMECANDO SVM")
for i in range(0, 10):
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'is_legendary')
	svmmodel = SVMModel(X_train, ytrain, X_test, ytest)
	res = svmmodel.kfold_accuracy()
	# print(res)
	res_svm.append(res)
f_pok = open("pok_svm_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
print("COMECANDO DECISION TREE")
for i in range(0, 10):
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'is_legendary')
	treemodel = DecisionTreeModel(X_train, ytrain, X_test, ytest)
	res_tree.append(treemodel.kfold_accuracy())
f_pok = open("pok_tree_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()

pre_processing = PreProcessing('../data/pd_speech_features.csv')
data = pre_processing.load_dataset()
X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'class')

print("COMECANDO O PD")
res_xgboost = []
print("COMECANDO XGBOOST")
for i in range(0, 10):
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'class')
	xgbmodel = XGBoostModel(X_train, ytrain, X_test, ytest)
	res_xgboost.append(xgbmodel.kfold_accuracy())
f_pok = open("pd_xgboost_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_xgboost[i])+"\n")
f_pok.close()

res_svm = []
print("COMECANDO SVM")
for i in range(0, 10):
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'class')
	svmmodel = SVMModel(X_train, ytrain, X_test, ytest)
	res_svm.append(svmmodel.kfold_accuracy())
f_pok = open("pd_svm_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_svm[i])+"\n")
f_pok.close()

res_tree = []
print("COMECANDO DECISION TREE")
for i in range(0, 10):
	X_train, X_test, ytrain, ytest = pre_processing.split_train_test(data, 'class')
	treemodel = DecisionTreeModel(X_train, ytrain, X_test, ytest)
	res_tree.append(treemodel.kfold_accuracy())
f_pok = open("pd_tree_def.txt","w+")
for i in range(0, 10):
     f_pok.write(str(res_tree[i])+"\n")
f_pok.close()

