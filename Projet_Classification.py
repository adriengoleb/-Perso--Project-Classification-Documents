#!/usr/bin/env python
# coding: utf-8

# ## Importation des librairies nécessaires

# In[2]:


import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import os
from collections import Counter
import re
from spellchecker import SpellChecker
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import spacy


# # Première Etude - premier data set

# ## Data Preparation

# In[168]:


#Partie traduction des données 

#lignes tres importantes sans lesquelles les librairies  ne marchent pas 
path = r"C:/Users/adrien/Desktop/EICNAM/2ème année/Echantillonage/Projet"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pop_path = "C:/Program Files/poppler-0.68.0/bin"


# In[169]:


#initialisation du correcteur python en français
import sys
spell = SpellChecker(language='fr')
nlp = spacy.load('fr_core_news_md')


# In[170]:


#initialisation du correcteur python en français
spell = SpellChecker(language='fr')


# In[171]:


#Création d'une liste de mot inutile
final_stopwords_list = list(fr_stop)


# In[172]:


data = pd.DataFrame(columns = ['Y', 'X'])
liste_fichier = os.listdir(path)
indexData = 0
for elem in liste_fichier :
    if elem[-4:] ==".pdf":
        print("\n fichier :" + elem)
        situation = elem[0]
        #permet de convertir les pdf en image
        pages = convert_from_path(elem, 200,poppler_path=pop_path) 
        liste_mot = []
        for i, page in enumerate(pages):
            #permet de traduire les mots identifer sur l'image en string
            text = pytesseract.image_to_string(page,lang='fra')
            text_minuscule = text.lower()
            text_minuscule = text_minuscule.replace('\n',' ')
            div = re.split('[^a-zA-Zà-î]', text_minuscule)
            liste_test = " ".join(div)
            #Lemnatisation : for token in nlp(text_minuscule) :
            for token in nlp(liste_test) :
                liste_mot.append(token.lemma_)
            l = []
            #Mettre for mot in "div" pour la div et "liste_mot" pour la lemmatisation
            for mot in div:
                #correction des mots mal traduits par pytesseract
                mot2 = spell.correction(mot)
                #on ne garde le mot que si il est plus long que deux lettres et qu'il n'appartient pas a la liste des stopwords
                if mot2 not in list(final_stopwords_list) and len(mot2)>1 :
                    l.append(mot2)
            ##Création du dataframe 
            text = " ".join(l)
            data.loc[indexData, "X"] = text
            data.loc[indexData, "Y"] = situation
            indexData += 1


# In[173]:


data


# In[174]:


print(data.info())


# In[175]:


data.to_csv(r'C:\Users\adrien\Desktop\EICNAM\2ème année\Echantillonage\Projet\data.csv', index=False)


# In[216]:


import pandas
df = pandas.read_csv('data.csv',encoding="utf-8")
df


# In[217]:


df['Y'] = df['Y'].replace('a','attestation hébergement')
df['Y'] = df['Y'].replace('b','bulletin de paie')
df['Y'] = df['Y'].replace('f','avis taxe foncière')
df['Y'] = df['Y'].replace('i','Fiche impot')
df['Y'] = df['Y'].replace('r','Relevé de compte')


# In[218]:


df


# In[219]:


df['Y'].value_counts()


# In[220]:


df.to_json (r'C:\Users\adrien\Desktop\EICNAM\2ème année\Echantillonage\Projet\df.json')


# In[221]:


import json

with open('df.json') as json_data:
    data_dict = json.load(json_data)

data_str = json.dumps(data_dict)

data_dict = json.loads(data_str)
print(data_dict)


# ## Features Engineering

# In[239]:


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer


# In[240]:


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(stop_words='english',analyzer='word',ngram_range=(1, 1), min_df=3) 


# In[242]:


df_words_train = vectorizer.fit_transform(df['X'])
df_words_train.shape


# In[243]:


print(vectorizer.stop_words) 


# In[244]:


print(vectorizer.stop_words_) 


# In[245]:


print ("Taille: {}",  len (vectorizer.vocabulary_))
print ("Contenu: {}", vectorizer.vocabulary_)


# In[247]:


df_words_train


# In[248]:


df_words_train = df_words_train.toarray()
print(df_words_train)


# In[249]:


# Create data frame
import pandas as pd
X=pd.DataFrame(df_words_train,columns=vectorizer.get_feature_names())
X


# ## Model Building

# In[250]:


y = df['Y']


# In[195]:


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
X_train_tf, X_test_tf, y_train_tf, y_test = train_test_split(X, y, test_size=0.20)


# In[196]:


clf = LinearSVC(random_state=42) # création d'un objet de type LinearSVC
clf.fit(X_train_tf, y_train_tf) # entrainement du modèle avec notre matrice Documents/Termes (pondération TF) et les classes


# In[197]:


## Création du jeu de test
pred = clf.predict(X_test_tf)
print(pred)

from sklearn.metrics import classification_report
#confusion matrix
print(classification_report(y_test,pred))


# ## CLassification Models

# ### Random Forest

# In[232]:


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model
rfc = RandomForestClassifier()


# In[233]:


# fit du model
rfc.fit(X_train_tf, y_train_tf)


# In[234]:


## Création du jeu de test
pred = rfc.predict(X_test_tf)
print(pred)

from sklearn.metrics import classification_report
#confusion matrix
print(classification_report(y_test,pred))


# ### Bayes Classifier

# In[235]:


# Entraînement d'un classificateur gaussien de Naive Bayes sur l'ensemble d'entraînement.
from sklearn.naive_bayes import GaussianNB


# In[236]:


#Instance du modèle
gnb = GaussianNB()


# In[237]:


# fit du model
gnb.fit(X_train_tf, y_train_tf)


# In[238]:


## Création du jeu de test
pred = gnb.predict(X_test_tf)
print(pred)

from sklearn.metrics import classification_report
#confusion matrix
print(classification_report(y_test,pred))


# In[ ]:





# # Nouvelle base d'entrainement - Data Set enrichi

# ### Feature Engineering - Data Preparation

# In[3]:


import json

json_df = pd.read_json('training_data_5_types.json', encoding='UTF-8')
json_df


# In[4]:


json_df['category'].value_counts()


# In[5]:


print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
#min_df : paramètre varié X3 : 10,40,70


vectorizer = CountVectorizer(stop_words='english',analyzer='word',ngram_range=(1, 1), min_df=40) 


# In[7]:


df_words_train = vectorizer.fit_transform(json_df['content'])
df_words_train.shape


# In[8]:


print(vectorizer.stop_words) 


# In[9]:


print(vectorizer.stop_words_) 


# In[10]:


print ("Taille: {}",  len (vectorizer.vocabulary_))


# In[11]:


df_words_train = df_words_train.toarray()
print(df_words_train)


# ### Définition Variables prédictives

# In[12]:


y = json_df['category']


# In[13]:


# Create data frame
import pandas as pd
X=pd.DataFrame(df_words_train,columns=vectorizer.get_feature_names())
X


# ### Construction bases apprentissages/test - Modèles de prédiction

# In[14]:


# Importer une fonction prévue pour séparer les sets.
from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import svm


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape


# In[17]:


from sklearn.model_selection import GridSearchCV


# ### SVM Modèle

# In[79]:


SVM = svm.SVC()


# In[80]:


parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),
              'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}


# In[81]:


clf_svm = GridSearchCV(SVM, parameters)


# In[82]:


# fit du model
clf_svm.fit(X_train, y_train)


# In[83]:


# Print the tuned parameters 
print("Tuned Decision Tree Parameters: {}".format(clf_svm.best_params_))


# In[84]:


#Affichage des prédictions
y_pred = clf_svm.predict(X_test)
y_pred


# In[85]:


df_results = pd.DataFrame({'y_Actual':y_test, 'y_Predicted':y_pred}, columns=['y_Actual', 'y_Predicted'])
df_results


# In[86]:


#Affichage d'un compte rendu des valeurs des métriques d'évaluation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(classification_report(y_test, y_pred))

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[87]:


#test Over Fitting ou non

print('Training set score: {:.4f}'.format(clf_svm.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_svm.score(X_test, y_test)))


# In[88]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[89]:


import seaborn as sns

results = {'y_Actual':y_test, 'y_Predicted':y_pred}
df_results = pd.DataFrame(results, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df_results['y_Actual'], df_results['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')


# ## Random Forest Modèle 

# In[90]:


# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model
rfc = RandomForestClassifier()


# In[91]:


from sklearn.model_selection import GridSearchCV


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [1,2,3,4,5,6,7,8,9,10],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


# In[92]:


# fit du model
CV_rfc.fit(X_train, y_train)


# In[93]:


# Print the tuned parameters 
print("Tuned Decision Tree Parameters: {}".format(clf_svm.best_params_))


# In[94]:


#Affichage des prédictions
y_pred = CV_rfc.predict(X_test)
y_pred


# In[95]:


df_results = pd.DataFrame({'y_Actual':y_test, 'y_Predicted':y_pred}, columns=['y_Actual', 'y_Predicted'])
df_results


# In[96]:


#Affichage d'un compte rendu des valeurs des métriques d'évaluation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(classification_report(y_test, y_pred))

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[97]:


#test Over Fitting ou non

print('Training set score: {:.4f}'.format(CV_rfc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(CV_rfc.score(X_test, y_test)))


# In[98]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[99]:


results = {'y_Actual':y_test, 'y_Predicted':y_pred}
df_results = pd.DataFrame(results, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df_results['y_Actual'], df_results['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')


# In[100]:


importance = CV_rfc.best_estimator_.feature_importances_
importance_df = pd.DataFrame(importance, index=X_train.columns, 
                      columns=["Importance"])
importance_df = importance_df.sort_values(by=['Importance'], ascending=False)
importance_df[:15]


# ### XGB - Gradient Boosting Classifier

# In[18]:


from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV  #Perforing grid search


# In[19]:


clf = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)


# In[20]:


parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}


# In[21]:


clf_xgb_cv = GridSearchCV(
    estimator=clf,
    param_grid=parameters,
)


# In[22]:


clf_xgb_cv.fit(X_train, y_train)


# In[23]:


# Print the tuned parameters 
print("Tuned Decision Tree Parameters: {}".format(clf_xgb_cv.best_params_))


# In[24]:


#Affichage des prédictions
y_pred = clf_xgb_cv.predict(X_test)
y_pred


# In[25]:


df_results = pd.DataFrame({'y_Actual':y_test, 'y_Predicted':y_pred}, columns=['y_Actual', 'y_Predicted'])
df_results


# In[26]:


#Affichage d'un compte rendu des valeurs des métriques d'évaluation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(classification_report(y_test, y_pred))

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[27]:


#test Over Fitting ou non

print('Training set score: {:.4f}'.format(clf_xgb_cv.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_xgb_cv.score(X_test, y_test)))


# In[28]:


import seaborn as sns

results = {'y_Actual':y_test, 'y_Predicted':y_pred}
df_results = pd.DataFrame(results, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df_results['y_Actual'], df_results['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')


# In[112]:


importance = clf_xgb_cv.best_estimator_.feature_importances_
importance_df = pd.DataFrame(importance, index=X_train.columns, 
                      columns=["Importance"])
importance_df = importance_df.sort_values(by=['Importance'], ascending=False)
importance_df[:15]


# ### MLP - Classifier Perceptron Multi-Couches

# In[114]:


mlp_gs = MLPClassifier(max_iter=100)


# In[115]:


import numpy as np

parameter_space = {
    'hidden_layer_sizes': np.arange(1, 10),
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


# In[116]:


CVclf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)


# In[117]:


CVclf.fit(X_train, y_train) # X is train samples and y is the corresponding labels


# In[118]:


# Print the tuned parameters 
print("Tuned Decision Tree Parameters: {}".format(CVclf.best_params_))


# In[119]:


#Affichage des prédictions
y_pred = CVclf.predict(X_test)
y_pred


# In[120]:


df_results = pd.DataFrame({'y_Actual':y_test, 'y_Predicted':y_pred}, columns=['y_Actual', 'y_Predicted'])
df_results


# In[121]:


#Affichage d'un compte rendu des valeurs des métriques d'évaluation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print(classification_report(y_test, y_pred))

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[122]:


#test Over Fitting ou non

print('Training set score: {:.4f}'.format(clf_xgb_cv.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_xgb_cv.score(X_test, y_test)))


# In[123]:


import seaborn as sns

results = {'y_Actual':y_test, 'y_Predicted':y_pred}
df_results = pd.DataFrame(results, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df_results['y_Actual'], df_results['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




