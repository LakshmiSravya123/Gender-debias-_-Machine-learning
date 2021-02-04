import pandas as pd
import numpy as np
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

nltk.download('punkt')
nltk.download('wordnet')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
stopwords = ['i','age','lol','rofl','haha','hehe','Mother','daughter','elder','grandfather','son','father','grandmother','uncle','Aunt','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","young","younger","old","older","student","class","girl","boy","yay","can","can't"]
data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/blogtext.csv",encoding="latin1")
data = pd.concat([data.age,data.text,data.topic,data.gender,data.sign,data.date],axis=1)
data=data[:30000]
#make some preperation for prediction, first change male->0 female ->1
#data.gender = [1 if each =="female" else 0 for each in data.gender]

data.dropna(axis=0,inplace=True)
data.head(5)

#Now prepare text of description data for prediction. Like, making lowercase, omitting unnecessary words,stopping words etc. 
description_list = []
lemma = nltk.WordNetLemmatizer()
data["des"] = data["text"].map(str) + data["topic"] + data["gender"]+data["sign"]+data["date"]
for description in data.des:
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)


max_features = 3000
count_vect = CountVectorizer(max_features = max_features,stop_words=stopwords)
matrix = count_vect.fit_transform(description_list).toarray()


y = data.iloc[:,0].values
x = matrix
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


rfc = RandomForestClassifier(n_estimators= 80, min_samples_split= 5, min_samples_leaf= 1, max_features='sqrt', max_depth=70, bootstrap= False)
rfc.fit(x_train,y_train)


y_pred = rfc.predict(x_test)

print("random accuracy: ",rfc.score(x_test,y_test))


clf = MultinomialNB(alpha=0.02).fit(x_train,y_train)
prediction=clf.predict(x_test)

print("multinobial accuracy: ",clf.score(x_test,y_test))


predictions_voting=[]
for i in range(len(prediction)):
    a=[prediction[i],y_pred[i]]
    predictions_voting.append(max(set(a), key=a.count))
print("manua voting Score -> ",accuracy_score(predictions_voting, y_test)*100)


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)

x=["Random Forest","Naive Bayes","SVM","Voting"]
plt.plot(x,[504,609,173,418],label="Before Bias Removal")
plt.plot(x,[603,610,224,505],label="After Bias Removal")
plt.plot(x,[572,572,572,572],label="Original")
plt.xlabel('Classifiers')
plt.ylabel('Female Counts')
plt.legend()
plt.show()

x=["Random Forest","Naive Bayes","SVM","Voting"]
plt.plot(x,[1057,952,1388,1143],label="Before Bias Removal")
plt.plot(x,[958,951,1337,1056],label="After Bias Removal")
plt.plot(x,[989,989,989,989],label="Original")
plt.xlabel('Classifiers')
plt.ylabel('Male Counts')
plt.legend()
plt.show()

x=["Random Forest","Naive Bayes","Voting"]
plt.plot(x,[69.67,57.34,64.23],label="Before Bias Removal")
plt.plot(x,[73.91,60.72,67.76],label="After Bias Removal")
plt.xlabel('Classifiers')
plt.ylabel('Accuracies')
plt.legend()
plt.show()
