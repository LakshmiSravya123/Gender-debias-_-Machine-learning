import pandas as pd
import numpy as np
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
stopwords = ['i','age','lol','rofl','haha','hehe','Mother','daughter','elder','grandfather','son','father','grandmother','uncle','Aunt','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","young","younger","old","older","student","class","girl","boy","yay","can","can't"]
data = pd.read_csv("/content/drive/My Drive/adult.csv",encoding="latin1")
data.columns = ['index', 'job', 'id', 'degree','no','marital-status','company','family-status','race','gender','n01','0','no2','country','salary']
data = pd.concat([data.gender,data.degree,data.job,data.company,data.race],axis=1)



data.dropna(axis=0,inplace=True)
data.head(5)

#Now prepare text of description data for prediction. Like, making lowercase, omitting unnecessary words,stopping words etc. 
description_list = []
lemma = nltk.WordNetLemmatizer()
data["des"] = data["degree"].map(str) + data["job"] + data["company"]+data["race"]

for description in data.des:
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

max_features = 3000
count_vect = CountVectorizer(max_features = max_features)
matrix = count_vect.fit_transform(description_list).toarray()


y = data.iloc[:,0].values
x = matrix
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


rfc = RandomForestClassifier(n_estimators= 90, min_samples_split= 2, min_samples_leaf= 2, max_features='auto', max_depth=None, bootstrap= True)
rfc.fit(x_train,y_train)


y_pred = rfc.predict(x_test)
print("random accuracy training: ",rfc.score(x_train,y_train))
print("random accuracy: ",rfc.score(x_test,y_test))


clf = MultinomialNB(alpha=41).fit(x_train,y_train)
prediction=clf.predict(x_test)
print("multinobial training: ",clf.score(x_train,y_train))
print("multinobial accuracy: ",clf.score(x_test,y_test))

clf_svc = SVC(kernel='rbf',C=1,gamma=0.1).fit(x_train,y_train)
prediction_svc=clf_svc.predict(x_test)
print("svc accuracy training: ",clf_svc.score(x_train,y_train))
print("SVC accuracy: ",clf_svc.score(x_test,y_test))


predictions_voting=[]
for i in range(len(prediction)):
    a=[prediction[i],y_pred[i],prediction_svc[i]]
    predictions_voting.append(max(set(a), key=a.count))
print("manual voting Score -> ",accuracy_score(predictions_voting, y_test)*100)



#Cs = [0.001, 0.01, 0.1, 1, 10]
#gammas = [0.001, 0.01, 0.1, 1]
#param_grid = {'C': Cs, 'gamma' : gammas}
#grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
#grid_search.fit(x_train, y_train)
#print(grid_search.best_params_)

#parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
#clf = GridSearchCV(MLPClassifier(), parameters)

#clf.fit(x_train, y_train)
#print(clf.score(x_train, y_train))
#print(clf.best_params_)

clf_mlp =MLPClassifier(hidden_layer_sizes=15,max_iter=2000).fit(x_train,y_train)
prediction_mlp=clf_mlp.predict(x_test)
print("mlp training: ",clf_mlp.score(x_train,y_train))
print("mlp accuracy: ",clf_mlp.score(x_test,y_test))

predictions_voting=[]
for i in range(len(prediction)):
    a=[prediction[i],y_pred[i],prediction_svc[i],prediction_mlp[i]]
    predictions_voting.append(max(set(a), key=a.count))
print("manual voting Score -> ",accuracy_score(predictions_voting, y_test)*100)

x=["Random Forest","Naive Bayes","SVM","MLP","Voting"]
plt.plot(x,[0.745085995085995,0.7174063267813268,0.7366016584766585,0.7401335995085995,0.7347825],label="Training")
plt.plot(x,[0.7271191646191646,0.7128378378378378,0.7277334152334153,0.7283476658476659,0.7301904176904176],label="Validation")

plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
