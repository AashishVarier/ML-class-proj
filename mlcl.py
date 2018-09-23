#!/home/littlegaintl/Codespace/MLclass/env python

# Compare LR,KNN,CART,NB,SVM Algorithms
import matplotlib.pyplot as plt
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load dataset1
#urll = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataframe = pandas.read_csv(urll, names=names)
#array = dataframe.values
#X = array[:,0:4]
#Y = array[:,4]
#dataset frequency descriptions
#print(dataframe.describe())
#display dataset box plots
#dataframe.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#def main()

 # load dataset2
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = pandas.read_csv(url, names=names)
array = dataset.values
X = array[:, :8]
Y = array[:, 8]
 #dataset
print(" No.of instance and Total attributes:") 
print(dataset.shape)
 #dataset frequency descriptions
print("Dataset frequency descriptions:")
print(dataset.describe())
 # histograms
dataset.hist()
plt.show()
# prepare configuration for cross validation
seed = 7
# preparation models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluation of each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Accuracy = %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#main()
