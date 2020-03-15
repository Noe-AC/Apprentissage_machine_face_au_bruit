import numpy as np
from sklearn import datasets
import matplotlib.pylab as plt
from PIL import Image # pour convertir un array (64,0) à une image (8,8)
from sklearn.decomposition import PCA
from matplotlib.pyplot import text
import pandas as pd
from sklearn.model_selection import train_test_split
# On met les erreurs à OFF
import warnings
warnings.simplefilter("ignore")
# Le modèles d'apprentissage machine utilisés :
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import math


# Le but ici est de créer des images des chiffres

# On se donne des données
digits = datasets.load_digits() # 1797 lignes, 64 pixels, pixels à valeurs en {0,1,2,...,15}, 
X = digits.data   # source, X[i][j] = image i, pixel j
y = digits.target # but,    y[i] = chiffre en {0,1,2,...,9}

# On normalise X :
n = len(X) # = 1797
id_number = []
X_normed = [] # une liste Python d'arrays numpy
for i in range(n):
	X_normed.append(np.array(X[i])/15) # on y met les pixels normalisés en [0.0000,1.0000]
	id_number.append(i)

"""
# Pour afficher les digits 0 à 9 sur trois lignes (donc 27 chiffres sur 3 lignes)
nb_horiz = 10
nb_vert = 3
# Visualisation des chiffres
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_horiz*nb_vert):
	ax = fig.add_subplot(nb_vert,nb_horiz,i+1,xticks=[],yticks=[])
	# On transforme en array d'images
	w,h=8,8 # images 8x8
	image = np.zeros((w, h))
	for j in range(8):
		image[j] = X_normed[i][8*j:8*j+8]
	img = Image.fromarray(image)
	ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
	#ax.imshow(X[i], cmap=plt.cm.binary, interpolation='nearest')
	ax.text(0,7,str(digits.target[i]))
plt.show()
"""

"""
# Autre manière d'afficher ça en grille, plus clair en termes des indices (i,j) = (ligne,colonne)
sigmas = [0,0.1,0.3,0.5,0.7,1,10]
nb_horiz = 10
nb_vert = len(sigmas)
mu = 0 # moyenne
size = 64 # 64 pixels
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_vert): # (i,j) = (ligne,colonne)
	for j in range(nb_horiz):
		# On ajoute du bruit au vecteur d'image
		np.random.seed(10*i+j) # on seed sur l'entier i
		numbers = X_normed[j] # on normalise de 0 à 1 les pixels
		sigma = sigmas[i]
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = bruit + numbers
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		# On transforme le vecteur de pixels en matrice de pixels
		w,h=8,8 # images 8x8
		image = np.zeros((w, h))
		for k in range(8):
			image[k] = numbers[8*k:8*k+8]
		# On transforme la matrice de pixels en image
		img = Image.fromarray(image)
		# On crée la case et on la met dans la grille
		ax = fig.add_subplot(nb_vert,nb_horiz,1+10*i+j,xticks=[],yticks=[])
		ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
		ax.text(0,7,str(j))
plt.show()
"""


"""
# Ici le but est d'afficher la même ligne de chiffres mais avec plusieurs degrés d'écart-types
w,h=8,8 # images 8x8 pixels
sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
nb_horiz = 10
nb_vert = len(sigmas)
mu = 0 # moyenne
size = 64 # 64 pixels
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_vert): # (i,j) = (ligne,colonne)
	for j in range(nb_horiz+1): # +1 pour la colonne où j'écris sigma
		if j==0: # pour afficher sigma
			image = np.zeros((w, h)) # on crée un carré de zéros
			img = Image.fromarray(image)
			ax = fig.add_subplot(nb_vert,nb_horiz+1,1+10*i+i,xticks=[],yticks=[])
			ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
			ax.text(0,4,"σ ="+ "%.1f"%sigmas[i])
		if j!=0: # les chiffres
			# On ajoute du bruit au vecteur d'image
			chiffre=j-1
			np.random.seed(10*i+chiffre) # on seed sur l'entier i
			numbers = X_normed[chiffre] # on normalise de 0 à 1 les pixels
			sigma = sigmas[i]
			bruit = np.random.normal(loc=mu,scale=sigma,size=size)
			numbers = bruit + numbers
			numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
			numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
			# On transforme le vecteur de pixels en matrice de pixels
			image = np.zeros((w, h))
			for k in range(8):
				image[k] = numbers[8*k:8*k+8]
			# On transforme la matrice de pixels en image
			img = Image.fromarray(image)
			# On crée la case et on la met dans la grille
			ax = fig.add_subplot(nb_vert,nb_horiz+1,1+10*i+j+i,xticks=[],yticks=[])
			ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
			ax.text(0,7,str(chiffre))
plt.show()
"""




"""
# PCA : Principal Component Analysis
pca = PCA(n_components=2)
X_noisy = [] # une liste Python d'arrays numpy
for i in range(n):

	X_normed[i] + bruit
	X_noisy.append(np.array(X[i])/15) # on y met les pixels normalisés en [0.0000,1.0000]

proj = pca.fit_transform(X_normed)
plt.scatter(proj[:,0],proj[:,1], c=y)
plt.colorbar()
plt.title('PCA pour X normalisé')
plt.show()
"""


"""
# Ici le but est d'afficher le PCA avec plusieurs degrés d'écart-types
w,h=8,8 # images 8x8 pixels
sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
nb_horiz = 4
nb_vert = 3
mu = 0 # moyenne
size = 64 # 64 pixels
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(nb_vert): # (i,j) = (ligne,colonne)
	for j in range(nb_horiz): # +1 pour la colonne où j'écris sigma
		indice = 4*i+j
		sigma = sigmas[indice]
		X_noisy = [] # une liste Python d'arrays numpy
		for k in range(n):
			np.random.seed(indice*n+k) # on seed sur l'entier i
			bruit = np.random.normal(loc=mu,scale=sigma,size=size)
			numbers = X_normed[k] + bruit
			numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
			numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
			X_noisy.append(numbers)
		# On crée un carré
		#image = np.zeros((w, h)) # on crée un carré de zéro
		#img = Image.fromarray(image) # on crée un carré blanc
		ax = fig.add_subplot(nb_vert,nb_horiz,1+4*i+j,xticks=[],yticks=[])
		#ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
		# Affichage du PCA
		pca = PCA(n_components=2)
		proj = pca.fit_transform(X_noisy)
		#ax.text(0,0,"σ ="+ "%.1f"%sigma)
		plt.scatter(proj[:,0],proj[:,1], c=y)
		if indice in [5,6,8]:plt.gca().invert_yaxis() # pour avoir les PCA orientés de la même manière (ici c'est un flip vertical de l'image)
		#plt.colorbar() # barre de couleurs
		text(x=0.15, y=1-0.05, s="σ="+ "%.1f"%sigma, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
plt.show()
"""

# Ok, maintenant que j'ai fait les PCA pour divers bruits, je peux passer à l'apprentissage machine.
# Deux possibilités :
"""
- soit on met du bruit juste sur le but
- soit on met du bruit sur la source et sur le but

On peut regarder s'il y a une grande différence entre les deux...
Bref, je dois faire les deux.
Puis je peux les comparer.

Avant de mettre du bruit, je vais commencer par regarder les scores d'apprensissage sans bruit, ainsi que la matrice de confusion
"""




# On met les features dans X
X = np.array(X_normed)
y = np.array(y)
# On découpe les données en train + test
test_size=0.25
random_state = 27
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X,y,id_number,test_size=test_size, random_state=random_state)
print(X.shape,X_train.shape,X_test.shape,y.shape,y_train.shape,y_test.shape) # donne : (1797, 64) (1347, 64) (450, 64) (1797,) (1347,) (450,)
# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.
# On regarde les différents classificateurs
clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train) # Apprentissage : KNN
print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = GaussianNB().fit(X_train,y_train) # Apprentissage : bayésien naïf gaussien
print("BNG :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = BernoulliNB().fit(X_train,y_train) # Apprentissage : bayésien naïf Bernoulli
print("BNB :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C).fit(X_train,y_train) # Apprentissage : SVM
print("SVM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=100).fit(X_train,y_train) # Apprentissage : régression logistique avec solveur 'lbfgs'
print("lbf :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=100).fit(X_train,y_train) # Apprentissage : régression logistique avec solveur 'liblinear'
print("lib :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = RandomForestClassifier().fit(X_train,y_train) # Apprentissage : Random Forest Classifier
print("RFC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = Perceptron().fit(X_train,y_train) # Apprentissage : Perceptron
print("Per :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 )) 
clf = SGDClassifier().fit(X_train,y_train) # Apprentissage : SGDClassifier, descente de gradient stochastique, version classificateur
print("SGD :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
clf = DecisionTreeClassifier().fit(X_train,y_train) # Apprentissage : DecisionTreeClassifier
print("DTC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))


####################################################################################################
####################################################################################################

"""

Prédictions :

KNN :	 train : 100.0%	 test : 98.2% <---
BNG :	 train : 87.6%	 test : 85.6%
BNB :	 train : 86.3%	 test : 87.3%
SVM :	 train : 99.2%	 test : 98.2% <---
lbf :	 train : 98.4%	 test : 96.2%
lib :	 train : 97.8%	 test : 96.7%
RFC :	 train : 100.0%	 test : 93.6%
Per :	 train : 97.2%	 test : 95.1%
SGD :	 train : 98.5%	 test : 96.0%
DTC :	 train : 100.0%	 test : 85.1%

KNN et SVM semblent aussi bons

"""


####################################################################################################
####################################################################################################

# Ici je vais faire une matrice de confusion
# Colonne est la réalité, ligne est la prédiction

clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C)
clf.fit(X_train,y_train)
y_true = y_test
y_pred = clf.predict(X_test)
matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
nb_predictions = len(y_true)
print("\nNombre de prédictions : ",nb_predictions)
score = clf.score(X_test,y_test)
print("\nNombre d'erreurs : ", int((1-score)*nb_predictions) )
print("\nMatrice de confusion pour SVM :")
print(matrix)


"""

Nombre de prédictions :  450
Nombre d'erreurs :  8
Matrice de confusion (i,j) = (réalité,prédit) :


Matrice de confusion pour SVM :

[[39  0  0  0  0  0  0  0  0  0]
 [ 0 37  0  0  0  0  0  0  0  0]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 45  0  1  0  0  0  0]
 [ 0  0  0  0 49  0  0  0  1  0]
 [ 0  0  0  0  0 48  1  0  0  0]
 [ 0  0  0  0  0  1 51  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  2  0  1  0  0  0  0 42  0]
 [ 0  0  0  0  0  1  0  0  0 50]]

"""

####################################################################################################
####################################################################################################

# Ici je trouve quelles images sont mal prédites

nombre_derreurs = 0
erreurs = []
for i in range(len(y_pred)):
	if y_true[i]!=y_pred[i]:
		nombre_derreurs += 1
		erreurs.append([int(id_test[i]),y_true[i],y_pred[i]])
		#print("i : ",i, "\ty_true : ",y_true[i],"\ty_pred : ",y_pred[i],"id_test =",int(id_test[i]))
print("Nombre d'erreurs =",nombre_derreurs)
#print(erreurs)
df_erreurs = pd.DataFrame(erreurs,columns=["id_test","y_true","y_pred"])
df_erreurs.sort_values(by=['id_test'], inplace=True)
id_erreurs = list(df_erreurs['id_test'])
print("\n",df_erreurs)

"""
    id_test  y_true  y_pred
6       37       9       5
7      792       6       5
1      905       8       1
2     1361       5       6
4     1491       8       3
0     1553       8       1
3     1628       4       8
5     1729       3       5
"""


"""
#plt.plot(id_erreurs)
binwidth = 30
plt.hist(id_erreurs,bins=range(min(id_erreurs), max(id_erreurs) + binwidth, binwidth))
plt.show()
"""

####################################################################################################
####################################################################################################

# On peut faire des plot et scatter de points

"""
# Isomap : Isometric mapping
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=1, n_components=2)
proj = iso.fit_transform(X)
plt.scatter(proj[:,0],proj[:,1],c=y)
plt.colorbar()
plt.show()
"""


"""
# PCA : Principal Component Analysis
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
plt.scatter(proj[:,0],proj[:,1], c=y)
plt.colorbar()
# On ne regarde pas les points qui sont trop loin
plt.xlim(-32, 20)
plt.ylim(-11, 21)
plt.title('Analyse en composantes principales (PCA)',size=16);
plt.show()
"""


####################################################################################################
####################################################################################################

"""
Maintenant que j'ai les scores d'apprentissage et la matrice de confusion et les erreurs je peux ajouter du bruit.
D'abord, on ajoute du bruit juste sur les données test, mais non sur les données train.
Le but ici est de regarder l'évolution des scores d'apprentissage pour le bruit qui va de sigma en les 12 valeurs [0.0, 0.1, 0.2, ..., 1.0, 1.1].
"""


"""
# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.
# On met les erreurs à OFF
# On regarde les différents classificateurs
# On regarde les différents classificateurs
size = 64 # 64 pixels
mu = 0 # moyenne
sigma_min  = 0.0
sigma_max  = 0.5
sigma_jump = 0.5/1000
sigmas = list(np.arange(sigma_min,sigma_max,sigma_jump))
#sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
n = len(X_test)
# On ajoute le bruit à y_test
scores_KNN = []
scores_BNG = []
scores_BNB = []
scores_SVM = []
scores_lbf = []
scores_lib = []
scores_RFC = []
scores_Per = []
scores_SGD = []
scores_DTC = []
for i in range(len(sigmas)):
	sigma = sigmas[i]
	X_test_noisy = [] # une liste Python d'arrays numpy
	for j in range(n):
		np.random.seed(i*n+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_test[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_test_noisy.append(numbers)
	print("\nσ = %.4f"%sigma)
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train) # Apprentissage : KNN
	print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_KNN.append(clf.score(X_test_noisy,y_test))
	clf = GaussianNB().fit(X_train,y_train) # Apprentissage : bayésien naïf gaussien
	print("BNG :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_BNG.append(clf.score(X_test_noisy,y_test))
	clf = BernoulliNB().fit(X_train,y_train) # Apprentissage : bayésien naïf Bernoulli
	print("BNB :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_BNB.append(clf.score(X_test_noisy,y_test))
	clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C).fit(X_train,y_train) # Apprentissage : SVM
	print("SVM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_SVM.append(clf.score(X_test_noisy,y_test))
	clf = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=100).fit(X_train,y_train) # Apprentissage : régression logistique avec solveur 'lbfgs'
	print("lbf :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_lbf.append(clf.score(X_test_noisy,y_test))
	clf = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=100).fit(X_train,y_train) # Apprentissage : régression logistique avec solveur 'liblinear'
	print("lib :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_lib.append(clf.score(X_test_noisy,y_test))
	clf = RandomForestClassifier().fit(X_train,y_train) # Apprentissage : Random Forest Classifier
	print("RFC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_RFC.append(clf.score(X_test_noisy,y_test))
	clf = Perceptron().fit(X_train,y_train) # Apprentissage : Perceptron
	print("Per :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 )) 
	scores_Per.append(clf.score(X_test_noisy,y_test))
	clf = SGDClassifier().fit(X_train,y_train) # Apprentissage : SGDClassifier, descente de gradient stochastique, version classificateur
	print("SGD :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_SGD.append(clf.score(X_test_noisy,y_test))
	clf = DecisionTreeClassifier().fit(X_train,y_train) # Apprentissage : DecisionTreeClassifier
	print("DTC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_DTC.append(clf.score(X_test_noisy,y_test))
# Maintenant on peut faire un plot :
plt.plot(sigmas,scores_KNN,label="KNN")
plt.plot(sigmas,scores_BNG,label="BNG")
plt.plot(sigmas,scores_BNB,label="BNB")
plt.plot(sigmas,scores_SVM,label="SVM")
plt.plot(sigmas,scores_lbf,label="lbf")
plt.plot(sigmas,scores_lib,label="lib")
plt.plot(sigmas,scores_RFC,label="RFC")
plt.plot(sigmas,scores_Per,label="Per")
plt.plot(sigmas,scores_SGD,label="SGD")
plt.plot(sigmas,scores_DTC,label="DTC")
plt.legend(loc='best')
plt.axis('tight')
plt.title("Scores d'apprentissage pour divers σ")
plt.xlabel("σ")
plt.ylabel("Scores")
plt.show()
"""


# Maintenant on peut fitter une courbe quadratique à travers le sommet pour KNN

"""
# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.
# On met les erreurs à OFF
# On regarde les différents classificateurs
# On regarde les différents classificateurs
size = 64 # 64 pixels
mu = 0 # moyenne
sigma_min  = 0.0
sigma_max  = 10
sigma_jump = sigma_max/1000
sigmas = list(np.arange(sigma_min,sigma_max,sigma_jump))
#sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
n = len(X_test)
# On se donne un classificateur KNN
clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train) # Apprentissage : KNN
# On ajoute le bruit à y_test
scores_KNN = []
for i in range(len(sigmas)):
	sigma = sigmas[i]
	X_test_noisy = [] # une liste Python d'arrays numpy
	for j in range(n):
		np.random.seed(i*n+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_test[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_test_noisy.append(numbers)
	print("i =",i,"\tσ = %.4f"%sigma,"KNN - train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_KNN.append(clf.score(X_test_noisy,y_test))
# Maintenant on peut faire un plot :
x = np.array(sigmas) # On met les sigmas sous forme d'array NumPy pour le fitting
y = np.array(scores_KNN) # On met les scores sous forme d'array NumPy pour le fitting
coeff = np.polyfit(x=x, y=y, deg=2, rcond=None, full=False, w=None, cov=False)
print(coeff) # [-1.31689519  0.31301857  0.96956187]
#coeff= [-0.5,0,0.982] # on ajuste les coeff à la main
#z = coeff[0]*x*x + coeff[1]*x + coeff[2]
# Non c'est pas quadratique
z = np.exp(-x*x/2)
string = "x = np.array(["
for i in range(len(x)):
	if i<len(x)-1:
		string+= "%.4f"%x[i] + ","
	if i==len(x)-1:
		string+= "%.4f"%x[i] + "])"
print(string)
string = "y = np.array(["
for i in range(len(y)):
	if i<len(y)-1:
		string+= "%.4f"%y[i] + ","
	if i==len(y)-1:
		string+= "%.4f"%y[i] + "])"
print(string)
z = (0.9822-0.1)/(1+4*x**4) + 0.1
plt.plot(x,y,label='KNN')
plt.plot(x,z,label='f(σ)=0.9822/(1+4σ⁴)')
plt.title("Scores pour KNN et graphe de f(σ)=0.9822/(1+4σ⁴)")
plt.legend(loc='best')
plt.axis('tight')
plt.xlabel("σ")
plt.ylabel("Scores")
plt.show()
"""



"""
Ok, maintenant j'ai tout fait pour le bruit ajouté juste au test
Je peux alors regarder pour le cas où on ajoute le bruit aussi à l'entraînement
"""

"""
# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.
# On met les erreurs à OFF
# On regarde les différents classificateurs
# On regarde les différents classificateurs
size = 64 # 64 pixels
mu = 0 # moyenne
sigma_min  = 0.0
sigma_max  = 3.0
sigma_jump = sigma_max/100
sigmas = list(np.arange(sigma_min,sigma_max,sigma_jump))
#sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
n = len(X_test)
# On ajoute le bruit à y_test
scores_KNN = []
scores_BNG = []
scores_BNB = []
scores_SVM = []
scores_lbf = []
scores_lib = []
scores_RFC = []
scores_Per = []
scores_SGD = []
scores_DTC = []
for i in range(len(sigmas)):
	sigma = sigmas[i]
	X_train_noisy = [] # une liste Python d'arrays numpy
	for j in range(len(X_train)):
		np.random.seed(i*len(X_train)+i*len(X_test)+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_train[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_train_noisy.append(numbers)
	X_test_noisy = [] # une liste Python d'arrays numpy
	for j in range(len(X_test)):
		np.random.seed((i+1)*len(X_train) + i*len(X_test)+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_test[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_test_noisy.append(numbers)
	print("\nσ = %.4f"%sigma)
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train_noisy,y_train) # Apprentissage : KNN
	print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_KNN.append(clf.score(X_test_noisy,y_test))
	
	clf = GaussianNB().fit(X_train_noisy,y_train) # Apprentissage : bayésien naïf gaussien
	print("BNG :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_BNG.append(clf.score(X_test_noisy,y_test))
	
	clf = BernoulliNB().fit(X_train_noisy,y_train) # Apprentissage : bayésien naïf Bernoulli
	print("BNB :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_BNB.append(clf.score(X_test_noisy,y_test))
	
	clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C).fit(X_train_noisy,y_train) # Apprentissage : SVM
	print("SVM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_SVM.append(clf.score(X_test_noisy,y_test))
	
	clf = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=100).fit(X_train_noisy,y_train) # Apprentissage : régression logistique avec solveur 'lbfgs'
	print("lbf :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_lbf.append(clf.score(X_test_noisy,y_test))
	
	clf = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=100).fit(X_train_noisy,y_train) # Apprentissage : régression logistique avec solveur 'liblinear'
	print("lib :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_lib.append(clf.score(X_test_noisy,y_test))
	
	clf = RandomForestClassifier().fit(X_train_noisy,y_train) # Apprentissage : Random Forest Classifier
	print("RFC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_RFC.append(clf.score(X_test_noisy,y_test))
	
	clf = Perceptron().fit(X_train_noisy,y_train) # Apprentissage : Perceptron
	print("Per :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 )) 
	scores_Per.append(clf.score(X_test_noisy,y_test))
	
	clf = SGDClassifier().fit(X_train_noisy,y_train) # Apprentissage : SGDClassifier, descente de gradient stochastique, version classificateur
	print("SGD :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_SGD.append(clf.score(X_test_noisy,y_test))
	
	clf = DecisionTreeClassifier().fit(X_train_noisy,y_train) # Apprentissage : DecisionTreeClassifier
	print("DTC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_DTC.append(clf.score(X_test_noisy,y_test))
# Maintenant on peut faire un plot :
plt.plot(sigmas,scores_KNN,label="KNN")
plt.plot(sigmas,scores_BNG,label="BNG")
plt.plot(sigmas,scores_BNB,label="BNB")
plt.plot(sigmas,scores_SVM,label="SVM")
plt.plot(sigmas,scores_lbf,label="lbf")
plt.plot(sigmas,scores_lib,label="lib")
plt.plot(sigmas,scores_RFC,label="RFC")
plt.plot(sigmas,scores_Per,label="Per")
plt.plot(sigmas,scores_SGD,label="SGD")
plt.plot(sigmas,scores_DTC,label="DTC")
plt.legend(loc='best')
plt.axis('tight')
plt.title("Scores d'apprentissage pour divers σ")
plt.xlabel("σ")
plt.ylabel("Scores")
plt.show()
"""


# Ici le gagnant semble être SVM.
"""
Pour la régression je vais donc me baser sur SVM.
On verra ce que ça donne.
En particulier je pourrais comparer X_train,X_test_noise de KNN avec X_train_noise,X_test_noise de SVM après ça.
"""


# On prend SVM
# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.
# On met les erreurs à OFF
# On regarde les différents classificateurs
# On regarde les différents classificateurs
size = 64 # 64 pixels
mu = 0 # moyenne
sigma_min  = 0.0
sigma_max  = 10.0
sigma_jump = sigma_max/1000
sigmas = list(np.arange(sigma_min,sigma_max,sigma_jump))
#sigmas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
n = len(X_test)
# On se donne un classificateur SVM
# On ajoute le bruit à y_test
scores_SVM = []
for i in range(len(sigmas)):
	sigma = sigmas[i]
	X_train_noisy = [] # une liste Python d'arrays numpy
	for j in range(len(X_train)):
		np.random.seed(i*len(X_train)+i*len(X_test)+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_train[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_train_noisy.append(numbers)
	X_test_noisy = [] # une liste Python d'arrays numpy
	for j in range(n):
		np.random.seed((i+1)*len(X_train)+i*len(X_test)+j)
		bruit = np.random.normal(loc=mu,scale=sigma,size=size)
		numbers = X_test[j] + bruit
		numbers = np.array([max(0,x) for x in numbers]) # Si un nombre est négatif on le met à 0
		numbers = np.array([min(1,x) for x in numbers]) # Si un nombre est > 1 on le met à 1
		X_test_noisy.append(numbers)
	clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C).fit(X_train_noisy,y_train) # Apprentissage : SVM
	print("i =",i,"\tσ = %.4f"%sigma,"SVM - train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train_noisy,y_train)*100 , clf.score(X_test_noisy,y_test)*100 ))
	scores_SVM.append(clf.score(X_test_noisy,y_test))
# Maintenant on peut faire un plot :
x = np.array(sigmas) # On met les sigmas sous forme d'array NumPy pour le fitting
y = np.array(scores_SVM) # On met les scores sous forme d'array NumPy pour le fitting
coeff = np.polyfit(x=x, y=y, deg=2, rcond=None, full=False, w=None, cov=False)
print(coeff) # [-1.31689519  0.31301857  0.96956187]
#coeff= [-0.5,0,0.982] # on ajuste les coeff à la main
#z = coeff[0]*x*x + coeff[1]*x + coeff[2]
# Non c'est pas quadratique
z = np.exp(-x*x/2)
string = "x = np.array(["
for i in range(len(x)):
	if i<len(x)-1:
		string+= "%.4f"%x[i] + ","
	if i==len(x)-1:
		string+= "%.4f"%x[i] + "])"
print(string)
string = "y = np.array(["
for i in range(len(y)):
	if i<len(y)-1:
		string+= "%.4f"%y[i] + ","
	if i==len(y)-1:
		string+= "%.4f"%y[i] + "])"
print(string)
z = (0.9822-0.1)/(1+4*x**4) + 0.1
plt.plot(x,y,label='SVM')
plt.plot(x,z,label='f(σ)=0.9822/(1+4σ⁴)')
plt.title("Scores pour SVM et graphe de f(σ)=0.9822/(1+4σ⁴)")
plt.legend(loc='best')
plt.axis('tight')
plt.xlabel("σ")
plt.ylabel("Scores")
plt.show()














"""

Les scores d'apprentissage pour divers bruits ajoutés à X_test mais non à X_train :

σ = 0.0
KNN :	 train : 100.0%	 test : 98.2%
BNG :	 train : 87.6%	 test : 85.3%
BNB :	 train : 86.3%	 test : 87.3%
SVM :	 train : 99.2%	 test : 98.2%
lbf :	 train : 98.4%	 test : 96.2%
lib :	 train : 97.8%	 test : 96.4%
RFC :	 train : 100.0%	 test : 93.8%
Per :	 train : 97.2%	 test : 95.1%
SGD :	 train : 98.2%	 test : 95.3%
DTC :	 train : 100.0%	 test : 83.6%

σ = 0.1
KNN :	 train : 100.0%	 test : 98.0%
BNG :	 train : 87.6%	 test : 14.7%
BNB :	 train : 86.3%	 test : 50.0%
SVM :	 train : 99.2%	 test : 97.8%
lbf :	 train : 98.4%	 test : 95.6%
lib :	 train : 97.8%	 test : 95.8%
RFC :	 train : 99.8%	 test : 88.2%
Per :	 train : 97.2%	 test : 94.0%
SGD :	 train : 98.1%	 test : 94.0%
DTC :	 train : 100.0%	 test : 70.4%

σ = 0.2
KNN :	 train : 100.0%	 test : 96.9%
BNG :	 train : 87.6%	 test : 9.8%
BNB :	 train : 86.3%	 test : 54.0%
SVM :	 train : 99.2%	 test : 94.9%
lbf :	 train : 98.4%	 test : 93.3%
lib :	 train : 97.8%	 test : 93.1%
RFC :	 train : 99.9%	 test : 81.6%
Per :	 train : 97.2%	 test : 88.2%
SGD :	 train : 97.1%	 test : 90.9%
DTC :	 train : 100.0%	 test : 54.4%

σ = 0.3
KNN :	 train : 100.0%	 test : 94.0%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 50.9%
SVM :	 train : 99.2%	 test : 89.8%
lbf :	 train : 98.4%	 test : 84.9%
lib :	 train : 97.8%	 test : 85.3%
RFC :	 train : 99.9%	 test : 68.2%
Per :	 train : 97.2%	 test : 79.6%
SGD :	 train : 97.0%	 test : 76.2%
DTC :	 train : 100.0%	 test : 43.3%

σ = 0.4
KNN :	 train : 100.0%	 test : 88.2%
BNG :	 train : 87.6%	 test : 13.3%
BNB :	 train : 86.3%	 test : 47.3%
SVM :	 train : 99.2%	 test : 81.3%
lbf :	 train : 98.4%	 test : 80.2%
lib :	 train : 97.8%	 test : 80.7%
RFC :	 train : 99.9%	 test : 56.2%
Per :	 train : 97.2%	 test : 69.8%
SGD :	 train : 98.4%	 test : 74.0%
DTC :	 train : 100.0%	 test : 38.9%

σ = 0.5
KNN :	 train : 100.0%	 test : 78.4%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 44.2%
SVM :	 train : 99.2%	 test : 68.7%
lbf :	 train : 98.4%	 test : 64.2%
lib :	 train : 97.8%	 test : 66.0%
RFC :	 train : 100.0%	 test : 49.3%
Per :	 train : 97.2%	 test : 58.4%
SGD :	 train : 97.8%	 test : 57.8%
DTC :	 train : 100.0%	 test : 30.0%

σ = 0.6
KNN :	 train : 100.0%	 test : 70.0%
BNG :	 train : 87.6%	 test : 13.6%
BNB :	 train : 86.3%	 test : 36.7%
SVM :	 train : 99.2%	 test : 59.1%
lbf :	 train : 98.4%	 test : 59.8%
lib :	 train : 97.8%	 test : 58.2%
RFC :	 train : 100.0%	 test : 35.8%
Per :	 train : 97.2%	 test : 51.3%
SGD :	 train : 97.6%	 test : 47.6%
DTC :	 train : 100.0%	 test : 26.9%

σ = 0.7
KNN :	 train : 100.0%	 test : 56.2%
BNG :	 train : 87.6%	 test : 10.2%
BNB :	 train : 86.3%	 test : 34.2%
SVM :	 train : 99.2%	 test : 51.3%
lbf :	 train : 98.4%	 test : 51.1%
lib :	 train : 97.8%	 test : 52.0%
RFC :	 train : 99.9%	 test : 31.6%
Per :	 train : 97.2%	 test : 43.8%
SGD :	 train : 98.1%	 test : 41.6%
DTC :	 train : 100.0%	 test : 21.8%

σ = 0.8
KNN :	 train : 100.0%	 test : 48.9%
BNG :	 train : 87.6%	 test : 9.1%
BNB :	 train : 86.3%	 test : 34.4%
SVM :	 train : 99.2%	 test : 48.4%
lbf :	 train : 98.4%	 test : 47.6%
lib :	 train : 97.8%	 test : 46.9%
RFC :	 train : 99.9%	 test : 30.4%
Per :	 train : 97.2%	 test : 38.7%
SGD :	 train : 96.2%	 test : 39.1%
DTC :	 train : 100.0%	 test : 21.1%

σ = 0.9
KNN :	 train : 100.0%	 test : 43.3%
BNG :	 train : 87.6%	 test : 8.7%
BNB :	 train : 86.3%	 test : 28.4%
SVM :	 train : 99.2%	 test : 39.3%
lbf :	 train : 98.4%	 test : 40.2%
lib :	 train : 97.8%	 test : 39.1%
RFC :	 train : 99.9%	 test : 28.2%
Per :	 train : 97.2%	 test : 34.2%
SGD :	 train : 97.8%	 test : 31.3%
DTC :	 train : 100.0%	 test : 21.6%

σ = 1.0
KNN :	 train : 100.0%	 test : 37.8%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 28.4%
SVM :	 train : 99.2%	 test : 35.1%
lbf :	 train : 98.4%	 test : 33.1%
lib :	 train : 97.8%	 test : 35.3%
RFC :	 train : 99.9%	 test : 23.3%
Per :	 train : 97.2%	 test : 31.3%
SGD :	 train : 97.9%	 test : 30.2%
DTC :	 train : 100.0%	 test : 18.2%

σ = 1.1
KNN :	 train : 100.0%	 test : 37.6%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 26.9%
SVM :	 train : 99.2%	 test : 34.4%
lbf :	 train : 98.4%	 test : 32.2%
lib :	 train : 97.8%	 test : 33.8%
RFC :	 train : 100.0%	 test : 24.7%
Per :	 train : 97.2%	 test : 28.2%
SGD :	 train : 96.8%	 test : 28.0%
DTC :	 train : 100.0%	 test : 20.4%

σ = 1.2
KNN :	 train : 100.0%	 test : 30.4%
BNG :	 train : 87.6%	 test : 10.7%
BNB :	 train : 86.3%	 test : 27.6%
SVM :	 train : 99.2%	 test : 33.1%
lbf :	 train : 98.4%	 test : 31.8%
lib :	 train : 97.8%	 test : 31.8%
RFC :	 train : 100.0%	 test : 23.6%
Per :	 train : 97.2%	 test : 28.4%
SGD :	 train : 95.8%	 test : 25.3%
DTC :	 train : 100.0%	 test : 19.6%

σ = 1.3
KNN :	 train : 100.0%	 test : 26.0%
BNG :	 train : 87.6%	 test : 8.9%
BNB :	 train : 86.3%	 test : 24.2%
SVM :	 train : 99.2%	 test : 27.8%
lbf :	 train : 98.4%	 test : 27.8%
lib :	 train : 97.8%	 test : 26.7%
RFC :	 train : 99.9%	 test : 18.7%
Per :	 train : 97.2%	 test : 22.0%
SGD :	 train : 98.4%	 test : 24.4%
DTC :	 train : 100.0%	 test : 13.1%

σ = 1.4
KNN :	 train : 100.0%	 test : 28.7%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 22.0%
SVM :	 train : 99.2%	 test : 23.8%
lbf :	 train : 98.4%	 test : 24.7%
lib :	 train : 97.8%	 test : 26.0%
RFC :	 train : 100.0%	 test : 20.7%
Per :	 train : 97.2%	 test : 22.4%
SGD :	 train : 98.6%	 test : 22.2%
DTC :	 train : 100.0%	 test : 17.6%

σ = 1.5
KNN :	 train : 100.0%	 test : 25.3%
BNG :	 train : 87.6%	 test : 12.7%
BNB :	 train : 86.3%	 test : 22.2%
SVM :	 train : 99.2%	 test : 25.3%
lbf :	 train : 98.4%	 test : 23.6%
lib :	 train : 97.8%	 test : 23.3%
RFC :	 train : 99.9%	 test : 19.3%
Per :	 train : 97.2%	 test : 19.8%
SGD :	 train : 98.2%	 test : 20.4%
DTC :	 train : 100.0%	 test : 14.9%

σ = 1.6
KNN :	 train : 100.0%	 test : 26.4%
BNG :	 train : 87.6%	 test : 11.6%
BNB :	 train : 86.3%	 test : 22.0%
SVM :	 train : 99.2%	 test : 22.7%
lbf :	 train : 98.4%	 test : 20.7%
lib :	 train : 97.8%	 test : 21.8%
RFC :	 train : 99.9%	 test : 17.1%
Per :	 train : 97.2%	 test : 20.4%
SGD :	 train : 98.1%	 test : 16.9%
DTC :	 train : 100.0%	 test : 16.0%

σ = 1.7
KNN :	 train : 100.0%	 test : 23.3%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 19.3%
SVM :	 train : 99.2%	 test : 24.4%
lbf :	 train : 98.4%	 test : 24.0%
lib :	 train : 97.8%	 test : 24.4%
RFC :	 train : 99.7%	 test : 15.6%
Per :	 train : 97.2%	 test : 23.1%
SGD :	 train : 96.9%	 test : 21.1%
DTC :	 train : 100.0%	 test : 13.3%

σ = 1.8
KNN :	 train : 100.0%	 test : 18.2%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 18.7%
SVM :	 train : 99.2%	 test : 19.8%
lbf :	 train : 98.4%	 test : 20.4%
lib :	 train : 97.8%	 test : 20.7%
RFC :	 train : 100.0%	 test : 19.1%
Per :	 train : 97.2%	 test : 18.7%
SGD :	 train : 97.8%	 test : 20.0%
DTC :	 train : 100.0%	 test : 11.8%

σ = 1.9
KNN :	 train : 100.0%	 test : 21.1%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 17.3%
SVM :	 train : 99.2%	 test : 21.1%
lbf :	 train : 98.4%	 test : 18.7%
lib :	 train : 97.8%	 test : 18.4%
RFC :	 train : 99.9%	 test : 16.4%
Per :	 train : 97.2%	 test : 18.7%
SGD :	 train : 97.3%	 test : 18.7%
DTC :	 train : 100.0%	 test : 13.3%

σ = 2.0
KNN :	 train : 100.0%	 test : 21.8%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 19.8%
SVM :	 train : 99.2%	 test : 20.7%
lbf :	 train : 98.4%	 test : 20.0%
lib :	 train : 97.8%	 test : 19.6%
RFC :	 train : 99.8%	 test : 15.8%
Per :	 train : 97.2%	 test : 19.1%
SGD :	 train : 98.2%	 test : 20.2%
DTC :	 train : 100.0%	 test : 14.2%

σ = 2.1
KNN :	 train : 100.0%	 test : 19.1%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 18.7%
SVM :	 train : 99.2%	 test : 20.7%
lbf :	 train : 98.4%	 test : 18.4%
lib :	 train : 97.8%	 test : 18.4%
RFC :	 train : 99.9%	 test : 16.4%
Per :	 train : 97.2%	 test : 15.8%
SGD :	 train : 98.1%	 test : 15.1%
DTC :	 train : 100.0%	 test : 14.2%

σ = 2.2
KNN :	 train : 100.0%	 test : 20.7%
BNG :	 train : 87.6%	 test : 11.6%
BNB :	 train : 86.3%	 test : 18.7%
SVM :	 train : 99.2%	 test : 21.1%
lbf :	 train : 98.4%	 test : 18.4%
lib :	 train : 97.8%	 test : 18.9%
RFC :	 train : 99.9%	 test : 16.4%
Per :	 train : 97.2%	 test : 19.3%
SGD :	 train : 97.6%	 test : 18.0%
DTC :	 train : 100.0%	 test : 14.2%

σ = 2.3
KNN :	 train : 100.0%	 test : 19.6%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 18.9%
SVM :	 train : 99.2%	 test : 19.1%
lbf :	 train : 98.4%	 test : 20.0%
lib :	 train : 97.8%	 test : 21.6%
RFC :	 train : 100.0%	 test : 16.4%
Per :	 train : 97.2%	 test : 21.8%
SGD :	 train : 97.6%	 test : 18.9%
DTC :	 train : 100.0%	 test : 13.6%

σ = 2.4
KNN :	 train : 100.0%	 test : 14.4%
BNG :	 train : 87.6%	 test : 9.6%
BNB :	 train : 86.3%	 test : 15.1%
SVM :	 train : 99.2%	 test : 18.7%
lbf :	 train : 98.4%	 test : 18.2%
lib :	 train : 97.8%	 test : 18.0%
RFC :	 train : 100.0%	 test : 14.9%
Per :	 train : 97.2%	 test : 15.6%
SGD :	 train : 98.6%	 test : 14.4%
DTC :	 train : 100.0%	 test : 14.0%

σ = 2.5
KNN :	 train : 100.0%	 test : 17.8%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 17.3%
SVM :	 train : 99.2%	 test : 16.7%
lbf :	 train : 98.4%	 test : 18.2%
lib :	 train : 97.8%	 test : 18.2%
RFC :	 train : 100.0%	 test : 14.2%
Per :	 train : 97.2%	 test : 19.1%
SGD :	 train : 98.1%	 test : 16.0%
DTC :	 train : 100.0%	 test : 12.2%

σ = 2.6
KNN :	 train : 100.0%	 test : 16.2%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 17.1%
SVM :	 train : 99.2%	 test : 14.2%
lbf :	 train : 98.4%	 test : 15.8%
lib :	 train : 97.8%	 test : 16.2%
RFC :	 train : 100.0%	 test : 12.7%
Per :	 train : 97.2%	 test : 16.0%
SGD :	 train : 97.3%	 test : 13.6%
DTC :	 train : 100.0%	 test : 11.6%

σ = 2.7
KNN :	 train : 100.0%	 test : 16.4%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 15.6%
SVM :	 train : 99.2%	 test : 19.3%
lbf :	 train : 98.4%	 test : 18.7%
lib :	 train : 97.8%	 test : 18.4%
RFC :	 train : 100.0%	 test : 16.9%
Per :	 train : 97.2%	 test : 16.7%
SGD :	 train : 98.1%	 test : 18.0%
DTC :	 train : 100.0%	 test : 9.6%

σ = 2.8
KNN :	 train : 100.0%	 test : 17.6%
BNG :	 train : 87.6%	 test : 9.6%
BNB :	 train : 86.3%	 test : 16.4%
SVM :	 train : 99.2%	 test : 18.4%
lbf :	 train : 98.4%	 test : 18.4%
lib :	 train : 97.8%	 test : 18.2%
RFC :	 train : 99.9%	 test : 13.6%
Per :	 train : 97.2%	 test : 14.7%
SGD :	 train : 98.1%	 test : 16.0%
DTC :	 train : 100.0%	 test : 14.2%

σ = 2.9
KNN :	 train : 100.0%	 test : 14.4%
BNG :	 train : 87.6%	 test : 9.8%
BNB :	 train : 86.3%	 test : 15.3%
SVM :	 train : 99.2%	 test : 14.4%
lbf :	 train : 98.4%	 test : 15.3%
lib :	 train : 97.8%	 test : 14.7%
RFC :	 train : 100.0%	 test : 14.2%
Per :	 train : 97.2%	 test : 14.0%
SGD :	 train : 98.0%	 test : 12.9%
DTC :	 train : 100.0%	 test : 10.4%

σ = 3.0
KNN :	 train : 100.0%	 test : 18.2%
BNG :	 train : 87.6%	 test : 8.4%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 17.3%
lbf :	 train : 98.4%	 test : 16.9%
lib :	 train : 97.8%	 test : 16.4%
RFC :	 train : 100.0%	 test : 12.2%
Per :	 train : 97.2%	 test : 15.6%
SGD :	 train : 98.1%	 test : 17.6%
DTC :	 train : 100.0%	 test : 10.7%

σ = 3.1
KNN :	 train : 100.0%	 test : 13.3%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 14.2%
SVM :	 train : 99.2%	 test : 12.4%
lbf :	 train : 98.4%	 test : 13.6%
lib :	 train : 97.8%	 test : 12.4%
RFC :	 train : 99.8%	 test : 13.8%
Per :	 train : 97.2%	 test : 12.4%
SGD :	 train : 97.7%	 test : 12.4%
DTC :	 train : 100.0%	 test : 13.3%

σ = 3.2
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 14.0%
lbf :	 train : 98.4%	 test : 14.2%
lib :	 train : 97.8%	 test : 16.2%
RFC :	 train : 100.0%	 test : 14.2%
Per :	 train : 97.2%	 test : 13.8%
SGD :	 train : 97.4%	 test : 12.9%
DTC :	 train : 100.0%	 test : 14.4%

σ = 3.3
KNN :	 train : 100.0%	 test : 17.6%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 17.1%
SVM :	 train : 99.2%	 test : 18.7%
lbf :	 train : 98.4%	 test : 19.3%
lib :	 train : 97.8%	 test : 17.3%
RFC :	 train : 99.9%	 test : 14.7%
Per :	 train : 97.2%	 test : 15.1%
SGD :	 train : 98.4%	 test : 15.1%
DTC :	 train : 100.0%	 test : 9.8%

σ = 3.4
KNN :	 train : 100.0%	 test : 15.1%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 16.4%
SVM :	 train : 99.2%	 test : 16.9%
lbf :	 train : 98.4%	 test : 17.8%
lib :	 train : 97.8%	 test : 16.4%
RFC :	 train : 99.9%	 test : 16.0%
Per :	 train : 97.2%	 test : 18.2%
SGD :	 train : 95.2%	 test : 17.1%
DTC :	 train : 100.0%	 test : 12.0%

σ = 3.5
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 15.3%
lbf :	 train : 98.4%	 test : 14.4%
lib :	 train : 97.8%	 test : 14.7%
RFC :	 train : 99.9%	 test : 11.6%
Per :	 train : 97.2%	 test : 15.6%
SGD :	 train : 98.1%	 test : 15.6%
DTC :	 train : 100.0%	 test : 12.2%

σ = 3.6
KNN :	 train : 100.0%	 test : 14.7%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 11.6%
SVM :	 train : 99.2%	 test : 14.0%
lbf :	 train : 98.4%	 test : 14.7%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 100.0%	 test : 12.4%
Per :	 train : 97.2%	 test : 12.7%
SGD :	 train : 97.8%	 test : 12.7%
DTC :	 train : 100.0%	 test : 12.0%

σ = 3.7
KNN :	 train : 100.0%	 test : 14.7%
BNG :	 train : 87.6%	 test : 10.7%
BNB :	 train : 86.3%	 test : 11.6%
SVM :	 train : 99.2%	 test : 12.9%
lbf :	 train : 98.4%	 test : 14.4%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 99.9%	 test : 12.0%
Per :	 train : 97.2%	 test : 13.1%
SGD :	 train : 97.4%	 test : 11.6%
DTC :	 train : 100.0%	 test : 11.8%

σ = 3.8
KNN :	 train : 100.0%	 test : 18.2%
BNG :	 train : 87.6%	 test : 8.4%
BNB :	 train : 86.3%	 test : 16.9%
SVM :	 train : 99.2%	 test : 17.3%
lbf :	 train : 98.4%	 test : 17.6%
lib :	 train : 97.8%	 test : 16.4%
RFC :	 train : 100.0%	 test : 15.1%
Per :	 train : 97.2%	 test : 14.7%
SGD :	 train : 97.6%	 test : 14.9%
DTC :	 train : 100.0%	 test : 12.7%

σ = 3.9
KNN :	 train : 100.0%	 test : 17.3%
BNG :	 train : 87.6%	 test : 10.7%
BNB :	 train : 86.3%	 test : 13.6%
SVM :	 train : 99.2%	 test : 13.3%
lbf :	 train : 98.4%	 test : 15.6%
lib :	 train : 97.8%	 test : 14.7%
RFC :	 train : 99.9%	 test : 13.1%
Per :	 train : 97.2%	 test : 13.6%
SGD :	 train : 95.9%	 test : 14.9%
DTC :	 train : 100.0%	 test : 11.3%

σ = 4.0
KNN :	 train : 100.0%	 test : 13.6%
BNG :	 train : 87.6%	 test : 9.3%
BNB :	 train : 86.3%	 test : 11.3%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 12.7%
RFC :	 train : 99.9%	 test : 9.1%
Per :	 train : 97.2%	 test : 11.3%
SGD :	 train : 97.7%	 test : 12.2%
DTC :	 train : 100.0%	 test : 9.8%

σ = 4.1
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 10.4%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 100.0%	 test : 12.9%
Per :	 train : 97.2%	 test : 12.7%
SGD :	 train : 97.9%	 test : 12.4%
DTC :	 train : 100.0%	 test : 12.7%

σ = 4.2
KNN :	 train : 100.0%	 test : 12.2%
BNG :	 train : 87.6%	 test : 8.4%
BNB :	 train : 86.3%	 test : 10.4%
SVM :	 train : 99.2%	 test : 12.0%
lbf :	 train : 98.4%	 test : 13.3%
lib :	 train : 97.8%	 test : 14.2%
RFC :	 train : 100.0%	 test : 10.9%
Per :	 train : 97.2%	 test : 12.7%
SGD :	 train : 98.9%	 test : 13.3%
DTC :	 train : 100.0%	 test : 10.2%

σ = 4.3
KNN :	 train : 100.0%	 test : 14.2%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 14.2%
SVM :	 train : 99.2%	 test : 12.9%
lbf :	 train : 98.4%	 test : 15.6%
lib :	 train : 97.8%	 test : 15.6%
RFC :	 train : 100.0%	 test : 12.9%
Per :	 train : 97.2%	 test : 14.9%
SGD :	 train : 97.6%	 test : 14.0%
DTC :	 train : 100.0%	 test : 13.3%

σ = 4.4
KNN :	 train : 100.0%	 test : 13.3%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 16.0%
lbf :	 train : 98.4%	 test : 14.2%
lib :	 train : 97.8%	 test : 14.2%
RFC :	 train : 99.9%	 test : 13.1%
Per :	 train : 97.2%	 test : 13.8%
SGD :	 train : 97.9%	 test : 15.1%
DTC :	 train : 100.0%	 test : 12.2%

σ = 4.5
KNN :	 train : 100.0%	 test : 13.6%
BNG :	 train : 87.6%	 test : 8.9%
BNB :	 train : 86.3%	 test : 13.6%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 12.9%
lib :	 train : 97.8%	 test : 14.4%
RFC :	 train : 99.9%	 test : 12.0%
Per :	 train : 97.2%	 test : 15.6%
SGD :	 train : 98.5%	 test : 13.6%
DTC :	 train : 100.0%	 test : 9.8%

σ = 4.6
KNN :	 train : 100.0%	 test : 14.7%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 12.0%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 12.7%
RFC :	 train : 99.9%	 test : 12.4%
Per :	 train : 97.2%	 test : 12.0%
SGD :	 train : 97.6%	 test : 12.2%
DTC :	 train : 100.0%	 test : 11.1%

σ = 4.7
KNN :	 train : 100.0%	 test : 14.9%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 12.0%
SVM :	 train : 99.2%	 test : 14.2%
lbf :	 train : 98.4%	 test : 15.6%
lib :	 train : 97.8%	 test : 16.0%
RFC :	 train : 99.9%	 test : 13.8%
Per :	 train : 97.2%	 test : 15.8%
SGD :	 train : 98.6%	 test : 14.9%
DTC :	 train : 100.0%	 test : 11.6%

σ = 4.8
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 10.4%
SVM :	 train : 99.2%	 test : 12.9%
lbf :	 train : 98.4%	 test : 12.4%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 99.9%	 test : 10.9%
Per :	 train : 97.2%	 test : 12.9%
SGD :	 train : 98.3%	 test : 14.4%
DTC :	 train : 100.0%	 test : 11.1%

σ = 4.9
KNN :	 train : 100.0%	 test : 11.1%
BNG :	 train : 87.6%	 test : 8.0%
BNB :	 train : 86.3%	 test : 12.0%
SVM :	 train : 99.2%	 test : 14.9%
lbf :	 train : 98.4%	 test : 14.4%
lib :	 train : 97.8%	 test : 15.3%
RFC :	 train : 99.9%	 test : 10.2%
Per :	 train : 97.2%	 test : 14.9%
SGD :	 train : 96.1%	 test : 13.6%
DTC :	 train : 100.0%	 test : 6.9%

σ = 5.0
KNN :	 train : 100.0%	 test : 13.6%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 12.7%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 100.0%	 test : 10.9%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 98.0%	 test : 12.7%
DTC :	 train : 100.0%	 test : 13.1%

σ = 5.1
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 11.6%
BNB :	 train : 86.3%	 test : 14.2%
SVM :	 train : 99.2%	 test : 14.9%
lbf :	 train : 98.4%	 test : 13.3%
lib :	 train : 97.8%	 test : 12.9%
RFC :	 train : 100.0%	 test : 9.3%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 97.9%	 test : 12.9%
DTC :	 train : 100.0%	 test : 11.3%

σ = 5.2
KNN :	 train : 100.0%	 test : 15.6%
BNG :	 train : 87.6%	 test : 12.7%
BNB :	 train : 86.3%	 test : 14.4%
SVM :	 train : 99.2%	 test : 14.0%
lbf :	 train : 98.4%	 test : 13.1%
lib :	 train : 97.8%	 test : 14.2%
RFC :	 train : 99.9%	 test : 15.3%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 96.6%	 test : 10.9%
DTC :	 train : 100.0%	 test : 11.6%

σ = 5.3
KNN :	 train : 100.0%	 test : 11.3%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 12.7%
SVM :	 train : 99.2%	 test : 12.0%
lbf :	 train : 98.4%	 test : 13.8%
lib :	 train : 97.8%	 test : 12.4%
RFC :	 train : 99.9%	 test : 11.3%
Per :	 train : 97.2%	 test : 10.9%
SGD :	 train : 97.7%	 test : 12.7%
DTC :	 train : 100.0%	 test : 11.8%

σ = 5.4
KNN :	 train : 100.0%	 test : 14.7%
BNG :	 train : 87.6%	 test : 9.8%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 14.0%
lbf :	 train : 98.4%	 test : 12.2%
lib :	 train : 97.8%	 test : 10.9%
RFC :	 train : 99.9%	 test : 12.9%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 98.0%	 test : 12.9%
DTC :	 train : 100.0%	 test : 11.3%

σ = 5.5
KNN :	 train : 100.0%	 test : 12.9%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 13.8%
SVM :	 train : 99.2%	 test : 12.4%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 12.0%
RFC :	 train : 100.0%	 test : 10.4%
Per :	 train : 97.2%	 test : 10.2%
SGD :	 train : 98.5%	 test : 10.4%
DTC :	 train : 100.0%	 test : 11.8%

σ = 5.6
KNN :	 train : 100.0%	 test : 13.6%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 9.3%
SVM :	 train : 99.2%	 test : 13.6%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 11.6%
RFC :	 train : 99.9%	 test : 9.6%
Per :	 train : 97.2%	 test : 12.2%
SGD :	 train : 97.8%	 test : 12.0%
DTC :	 train : 100.0%	 test : 9.8%

σ = 5.7
KNN :	 train : 100.0%	 test : 11.6%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 9.8%
SVM :	 train : 99.2%	 test : 9.3%
lbf :	 train : 98.4%	 test : 7.6%
lib :	 train : 97.8%	 test : 9.1%
RFC :	 train : 100.0%	 test : 8.9%
Per :	 train : 97.2%	 test : 8.7%
SGD :	 train : 97.1%	 test : 9.8%
DTC :	 train : 100.0%	 test : 11.3%

σ = 5.8
KNN :	 train : 100.0%	 test : 15.8%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 14.0%
SVM :	 train : 99.2%	 test : 13.6%
lbf :	 train : 98.4%	 test : 13.6%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 99.9%	 test : 12.9%
Per :	 train : 97.2%	 test : 13.1%
SGD :	 train : 98.2%	 test : 12.2%
DTC :	 train : 100.0%	 test : 12.2%

σ = 5.9
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 11.8%
BNB :	 train : 86.3%	 test : 12.4%
SVM :	 train : 99.2%	 test : 10.4%
lbf :	 train : 98.4%	 test : 12.0%
lib :	 train : 97.8%	 test : 12.9%
RFC :	 train : 100.0%	 test : 12.2%
Per :	 train : 97.2%	 test : 13.8%
SGD :	 train : 98.1%	 test : 11.6%
DTC :	 train : 100.0%	 test : 10.4%

σ = 6.0
KNN :	 train : 100.0%	 test : 11.1%
BNG :	 train : 87.6%	 test : 10.2%
BNB :	 train : 86.3%	 test : 13.3%
SVM :	 train : 99.2%	 test : 12.9%
lbf :	 train : 98.4%	 test : 10.7%
lib :	 train : 97.8%	 test : 10.9%
RFC :	 train : 99.9%	 test : 10.2%
Per :	 train : 97.2%	 test : 12.0%
SGD :	 train : 97.7%	 test : 11.8%
DTC :	 train : 100.0%	 test : 10.2%

σ = 6.1
KNN :	 train : 100.0%	 test : 13.3%
BNG :	 train : 87.6%	 test : 9.3%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 12.4%
lib :	 train : 97.8%	 test : 14.2%
RFC :	 train : 100.0%	 test : 12.4%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 97.6%	 test : 10.7%
DTC :	 train : 100.0%	 test : 10.2%

σ = 6.2
KNN :	 train : 100.0%	 test : 14.9%
BNG :	 train : 87.6%	 test : 10.2%
BNB :	 train : 86.3%	 test : 11.1%
SVM :	 train : 99.2%	 test : 13.6%
lbf :	 train : 98.4%	 test : 12.9%
lib :	 train : 97.8%	 test : 12.9%
RFC :	 train : 100.0%	 test : 14.4%
Per :	 train : 97.2%	 test : 12.9%
SGD :	 train : 98.4%	 test : 12.0%
DTC :	 train : 100.0%	 test : 12.2%

σ = 6.3
KNN :	 train : 100.0%	 test : 12.9%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 10.7%
SVM :	 train : 99.2%	 test : 13.6%
lbf :	 train : 98.4%	 test : 12.9%
lib :	 train : 97.8%	 test : 12.0%
RFC :	 train : 100.0%	 test : 11.1%
Per :	 train : 97.2%	 test : 9.6%
SGD :	 train : 97.6%	 test : 10.7%
DTC :	 train : 100.0%	 test : 12.0%

σ = 6.4
KNN :	 train : 100.0%	 test : 10.2%
BNG :	 train : 87.6%	 test : 14.0%
BNB :	 train : 86.3%	 test : 9.8%
SVM :	 train : 99.2%	 test : 10.7%
lbf :	 train : 98.4%	 test : 11.1%
lib :	 train : 97.8%	 test : 9.3%
RFC :	 train : 99.9%	 test : 10.0%
Per :	 train : 97.2%	 test : 10.2%
SGD :	 train : 98.4%	 test : 11.1%
DTC :	 train : 100.0%	 test : 11.8%

σ = 6.5
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 12.2%
SVM :	 train : 99.2%	 test : 13.8%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 12.7%
RFC :	 train : 99.9%	 test : 13.3%
Per :	 train : 97.2%	 test : 12.4%
SGD :	 train : 97.6%	 test : 11.6%
DTC :	 train : 100.0%	 test : 15.3%

σ = 6.6
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 11.6%
SVM :	 train : 99.2%	 test : 14.0%
lbf :	 train : 98.4%	 test : 13.1%
lib :	 train : 97.8%	 test : 13.6%
RFC :	 train : 100.0%	 test : 11.6%
Per :	 train : 97.2%	 test : 14.0%
SGD :	 train : 98.0%	 test : 12.9%
DTC :	 train : 100.0%	 test : 11.6%

σ = 6.7
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 12.4%
SVM :	 train : 99.2%	 test : 9.6%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 99.9%	 test : 10.7%
Per :	 train : 97.2%	 test : 11.6%
SGD :	 train : 98.3%	 test : 11.8%
DTC :	 train : 100.0%	 test : 10.4%

σ = 6.8
KNN :	 train : 100.0%	 test : 10.2%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 14.0%
SVM :	 train : 99.2%	 test : 13.1%
lbf :	 train : 98.4%	 test : 13.3%
lib :	 train : 97.8%	 test : 14.2%
RFC :	 train : 100.0%	 test : 11.6%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 98.3%	 test : 14.0%
DTC :	 train : 100.0%	 test : 6.9%

σ = 6.9
KNN :	 train : 100.0%	 test : 11.3%
BNG :	 train : 87.6%	 test : 12.9%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 13.3%
lbf :	 train : 98.4%	 test : 15.3%
lib :	 train : 97.8%	 test : 14.9%
RFC :	 train : 99.9%	 test : 10.0%
Per :	 train : 97.2%	 test : 15.6%
SGD :	 train : 98.0%	 test : 15.1%
DTC :	 train : 100.0%	 test : 11.8%

σ = 7.0
KNN :	 train : 100.0%	 test : 9.8%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 8.0%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 9.8%
lib :	 train : 97.8%	 test : 10.0%
RFC :	 train : 99.9%	 test : 9.3%
Per :	 train : 97.2%	 test : 9.1%
SGD :	 train : 95.7%	 test : 9.1%
DTC :	 train : 100.0%	 test : 9.8%

σ = 7.1
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 9.3%
BNB :	 train : 86.3%	 test : 10.7%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 12.7%
RFC :	 train : 99.9%	 test : 11.6%
Per :	 train : 97.2%	 test : 13.6%
SGD :	 train : 98.3%	 test : 12.7%
DTC :	 train : 100.0%	 test : 12.0%

σ = 7.2
KNN :	 train : 100.0%	 test : 10.2%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 12.2%
RFC :	 train : 100.0%	 test : 11.6%
Per :	 train : 97.2%	 test : 11.6%
SGD :	 train : 98.0%	 test : 10.7%
DTC :	 train : 100.0%	 test : 11.1%

σ = 7.3
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 9.8%
BNB :	 train : 86.3%	 test : 14.2%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 100.0%	 test : 14.2%
Per :	 train : 97.2%	 test : 12.9%
SGD :	 train : 97.8%	 test : 11.8%
DTC :	 train : 100.0%	 test : 11.8%

σ = 7.4
KNN :	 train : 100.0%	 test : 12.2%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 12.2%
SVM :	 train : 99.2%	 test : 11.6%
lbf :	 train : 98.4%	 test : 11.6%
lib :	 train : 97.8%	 test : 11.6%
RFC :	 train : 100.0%	 test : 10.7%
Per :	 train : 97.2%	 test : 12.0%
SGD :	 train : 97.7%	 test : 11.6%
DTC :	 train : 100.0%	 test : 11.6%

σ = 7.5
KNN :	 train : 100.0%	 test : 14.0%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 9.8%
SVM :	 train : 99.2%	 test : 12.2%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 13.6%
RFC :	 train : 99.9%	 test : 12.7%
Per :	 train : 97.2%	 test : 12.0%
SGD :	 train : 97.6%	 test : 11.1%
DTC :	 train : 100.0%	 test : 13.1%

σ = 7.6
KNN :	 train : 100.0%	 test : 9.3%
BNG :	 train : 87.6%	 test : 11.6%
BNB :	 train : 86.3%	 test : 12.2%
SVM :	 train : 99.2%	 test : 12.0%
lbf :	 train : 98.4%	 test : 14.0%
lib :	 train : 97.8%	 test : 12.9%
RFC :	 train : 99.9%	 test : 15.1%
Per :	 train : 97.2%	 test : 14.2%
SGD :	 train : 96.3%	 test : 13.6%
DTC :	 train : 100.0%	 test : 13.3%

σ = 7.7
KNN :	 train : 100.0%	 test : 11.3%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 9.6%
SVM :	 train : 99.2%	 test : 8.9%
lbf :	 train : 98.4%	 test : 10.9%
lib :	 train : 97.8%	 test : 11.3%
RFC :	 train : 100.0%	 test : 10.7%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 98.8%	 test : 12.9%
DTC :	 train : 100.0%	 test : 10.9%

σ = 7.8
KNN :	 train : 100.0%	 test : 11.6%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 11.3%
SVM :	 train : 99.2%	 test : 13.1%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 11.1%
RFC :	 train : 99.9%	 test : 11.6%
Per :	 train : 97.2%	 test : 12.9%
SGD :	 train : 97.0%	 test : 11.8%
DTC :	 train : 100.0%	 test : 9.8%

σ = 7.9
KNN :	 train : 100.0%	 test : 14.7%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 12.7%
SVM :	 train : 99.2%	 test : 13.1%
lbf :	 train : 98.4%	 test : 12.9%
lib :	 train : 97.8%	 test : 12.2%
RFC :	 train : 99.9%	 test : 13.6%
Per :	 train : 97.2%	 test : 13.6%
SGD :	 train : 97.4%	 test : 13.3%
DTC :	 train : 100.0%	 test : 12.2%

σ = 8.0
KNN :	 train : 100.0%	 test : 12.2%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 10.9%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 11.6%
RFC :	 train : 100.0%	 test : 10.7%
Per :	 train : 97.2%	 test : 11.6%
SGD :	 train : 97.4%	 test : 10.2%
DTC :	 train : 100.0%	 test : 8.4%

σ = 8.1
KNN :	 train : 100.0%	 test : 10.9%
BNG :	 train : 87.6%	 test : 8.7%
BNB :	 train : 86.3%	 test : 10.7%
SVM :	 train : 99.2%	 test : 11.1%
lbf :	 train : 98.4%	 test : 12.7%
lib :	 train : 97.8%	 test : 12.7%
RFC :	 train : 99.9%	 test : 9.6%
Per :	 train : 97.2%	 test : 12.4%
SGD :	 train : 97.3%	 test : 11.6%
DTC :	 train : 100.0%	 test : 12.2%

σ = 8.2
KNN :	 train : 100.0%	 test : 10.9%
BNG :	 train : 87.6%	 test : 9.3%
BNB :	 train : 86.3%	 test : 11.8%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 13.1%
lib :	 train : 97.8%	 test : 13.6%
RFC :	 train : 99.9%	 test : 10.2%
Per :	 train : 97.2%	 test : 13.1%
SGD :	 train : 97.6%	 test : 10.9%
DTC :	 train : 100.0%	 test : 10.2%

σ = 8.3
KNN :	 train : 100.0%	 test : 11.6%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 10.9%
SVM :	 train : 99.2%	 test : 13.8%
lbf :	 train : 98.4%	 test : 15.6%
lib :	 train : 97.8%	 test : 16.9%
RFC :	 train : 100.0%	 test : 14.0%
Per :	 train : 97.2%	 test : 13.6%
SGD :	 train : 97.3%	 test : 12.9%
DTC :	 train : 100.0%	 test : 11.8%

σ = 8.4
KNN :	 train : 100.0%	 test : 9.6%
BNG :	 train : 87.6%	 test : 8.2%
BNB :	 train : 86.3%	 test : 11.3%
SVM :	 train : 99.2%	 test : 9.6%
lbf :	 train : 98.4%	 test : 11.8%
lib :	 train : 97.8%	 test : 11.6%
RFC :	 train : 100.0%	 test : 11.3%
Per :	 train : 97.2%	 test : 11.1%
SGD :	 train : 97.8%	 test : 12.0%
DTC :	 train : 100.0%	 test : 10.4%

σ = 8.5
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 11.3%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 12.0%
lib :	 train : 97.8%	 test : 11.8%
RFC :	 train : 99.9%	 test : 13.3%
Per :	 train : 97.2%	 test : 10.4%
SGD :	 train : 98.6%	 test : 12.7%
DTC :	 train : 100.0%	 test : 9.1%

σ = 8.6
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 12.7%
SVM :	 train : 99.2%	 test : 12.4%
lbf :	 train : 98.4%	 test : 12.2%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 100.0%	 test : 11.8%
Per :	 train : 97.2%	 test : 11.6%
SGD :	 train : 96.1%	 test : 12.0%
DTC :	 train : 100.0%	 test : 11.1%

σ = 8.7
KNN :	 train : 100.0%	 test : 10.9%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 10.9%
SVM :	 train : 99.2%	 test : 12.0%
lbf :	 train : 98.4%	 test : 12.2%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 99.9%	 test : 9.8%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 95.6%	 test : 12.7%
DTC :	 train : 100.0%	 test : 11.1%

σ = 8.8
KNN :	 train : 100.0%	 test : 11.8%
BNG :	 train : 87.6%	 test : 12.4%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 13.3%
lbf :	 train : 98.4%	 test : 13.1%
lib :	 train : 97.8%	 test : 12.9%
RFC :	 train : 100.0%	 test : 11.8%
Per :	 train : 97.2%	 test : 11.6%
SGD :	 train : 97.9%	 test : 12.0%
DTC :	 train : 100.0%	 test : 9.3%

σ = 8.9
KNN :	 train : 100.0%	 test : 13.1%
BNG :	 train : 87.6%	 test : 9.6%
BNB :	 train : 86.3%	 test : 13.3%
SVM :	 train : 99.2%	 test : 10.9%
lbf :	 train : 98.4%	 test : 11.1%
lib :	 train : 97.8%	 test : 13.3%
RFC :	 train : 99.9%	 test : 10.9%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 97.9%	 test : 11.8%
DTC :	 train : 100.0%	 test : 12.7%

σ = 9.0
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 11.1%
BNB :	 train : 86.3%	 test : 12.2%
SVM :	 train : 99.2%	 test : 12.7%
lbf :	 train : 98.4%	 test : 12.0%
lib :	 train : 97.8%	 test : 14.0%
RFC :	 train : 100.0%	 test : 10.7%
Per :	 train : 97.2%	 test : 12.9%
SGD :	 train : 98.1%	 test : 12.9%
DTC :	 train : 100.0%	 test : 13.6%

σ = 9.1
KNN :	 train : 100.0%	 test : 11.8%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 12.9%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 12.2%
lib :	 train : 97.8%	 test : 11.6%
RFC :	 train : 99.9%	 test : 10.9%
Per :	 train : 97.2%	 test : 10.9%
SGD :	 train : 97.8%	 test : 10.2%
DTC :	 train : 100.0%	 test : 13.3%

σ = 9.2
KNN :	 train : 100.0%	 test : 10.4%
BNG :	 train : 87.6%	 test : 10.4%
BNB :	 train : 86.3%	 test : 10.7%
SVM :	 train : 99.2%	 test : 11.6%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 10.4%
RFC :	 train : 100.0%	 test : 10.7%
Per :	 train : 97.2%	 test : 9.3%
SGD :	 train : 97.7%	 test : 12.0%
DTC :	 train : 100.0%	 test : 12.0%

σ = 9.3
KNN :	 train : 100.0%	 test : 13.8%
BNG :	 train : 87.6%	 test : 12.0%
BNB :	 train : 86.3%	 test : 12.2%
SVM :	 train : 99.2%	 test : 12.2%
lbf :	 train : 98.4%	 test : 14.7%
lib :	 train : 97.8%	 test : 14.0%
RFC :	 train : 99.9%	 test : 13.8%
Per :	 train : 97.2%	 test : 14.7%
SGD :	 train : 96.1%	 test : 16.0%
DTC :	 train : 100.0%	 test : 14.4%

σ = 9.4
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 10.7%
BNB :	 train : 86.3%	 test : 11.6%
SVM :	 train : 99.2%	 test : 9.8%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 11.3%
RFC :	 train : 99.7%	 test : 10.9%
Per :	 train : 97.2%	 test : 10.9%
SGD :	 train : 97.3%	 test : 11.6%
DTC :	 train : 100.0%	 test : 11.3%

σ = 9.5
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 10.0%
BNB :	 train : 86.3%	 test : 8.9%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 11.3%
lib :	 train : 97.8%	 test : 11.1%
RFC :	 train : 100.0%	 test : 12.4%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 98.2%	 test : 10.7%
DTC :	 train : 100.0%	 test : 9.6%

σ = 9.6
KNN :	 train : 100.0%	 test : 12.4%
BNG :	 train : 87.6%	 test : 10.9%
BNB :	 train : 86.3%	 test : 10.0%
SVM :	 train : 99.2%	 test : 11.3%
lbf :	 train : 98.4%	 test : 12.0%
lib :	 train : 97.8%	 test : 12.2%
RFC :	 train : 99.9%	 test : 14.7%
Per :	 train : 97.2%	 test : 13.6%
SGD :	 train : 98.4%	 test : 11.6%
DTC :	 train : 100.0%	 test : 9.8%

σ = 9.7
KNN :	 train : 100.0%	 test : 12.0%
BNG :	 train : 87.6%	 test : 12.2%
BNB :	 train : 86.3%	 test : 10.7%
SVM :	 train : 99.2%	 test : 13.6%
lbf :	 train : 98.4%	 test : 13.8%
lib :	 train : 97.8%	 test : 14.0%
RFC :	 train : 100.0%	 test : 10.4%
Per :	 train : 97.2%	 test : 15.3%
SGD :	 train : 98.4%	 test : 15.3%
DTC :	 train : 100.0%	 test : 11.8%

σ = 9.8
KNN :	 train : 100.0%	 test : 14.0%
BNG :	 train : 87.6%	 test : 11.6%
BNB :	 train : 86.3%	 test : 13.1%
SVM :	 train : 99.2%	 test : 15.6%
lbf :	 train : 98.4%	 test : 14.9%
lib :	 train : 97.8%	 test : 14.4%
RFC :	 train : 99.9%	 test : 12.7%
Per :	 train : 97.2%	 test : 13.3%
SGD :	 train : 98.1%	 test : 13.3%
DTC :	 train : 100.0%	 test : 12.7%

σ = 9.9
KNN :	 train : 100.0%	 test : 13.6%
BNG :	 train : 87.6%	 test : 11.3%
BNB :	 train : 86.3%	 test : 11.6%
SVM :	 train : 99.2%	 test : 12.2%
lbf :	 train : 98.4%	 test : 14.2%
lib :	 train : 97.8%	 test : 13.1%
RFC :	 train : 99.9%	 test : 9.8%
Per :	 train : 97.2%	 test : 11.8%
SGD :	 train : 98.3%	 test : 12.4%
DTC :	 train : 100.0%	 test : 10.0%

"""











