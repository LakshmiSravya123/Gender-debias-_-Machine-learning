import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
nltk.download('punkt')
nltk.download('wordnet')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from google.colab import drive
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
drive.mount('/content/drive')
#stopwords = ['spinster','maternal','heroine','hostess','female','feminine','actress','empress','heroine','hostess','waitress',
#'landlady','firewoman','policewoman','firewomen','policewomen','stewardess','Boy','He','Father','Mother','Sister','Daughter','Brother','Uncle','Aunt','Him', 'His', 'Mr.','Men', 'Man', 'Gregarious', 'cautious', 'affable', 'amiable', 'avuncular', 'funniest', 'good-natured', 'jovial', 'likable', 'mild-mannered', 'personable', 'cruel', 'dour', 'insufferable', 'braver', 'humane', 'law-worthy', 'patient', 'sincere', 'tolerant', 'trustworthy', 'truthful', 'upstanding', 'anxious', 'insane', 'astute', 'scholarly', 'self-educated', 'ignorant','She', 'Girl', 'Her', 'Mrs.','Miss.', 'Woman', 'lady', 'Women', 'Bossy', 'Chattering','Gossiping', 'Submissive', 'Bitchy','Hysterical', 'Weeping', 'filly', 'biddy','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/gender-classifier-DFE-791531.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)
data=data[(data['gender']!="unknown")] 

#make some preperation for prediction, first change male->0 female ->1
data.gender = [1 if each =="female" else 0 for each in data.gender]
data.dropna(axis=0,inplace=True)
data.head(5)

#Now prepare text of description data for prediction. Like, making lowercase, omitting unnecessary words,stopping words etc. 
description_list = []
lemma = nltk.WordNetLemmatizer()
for description in data.description:
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


#parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
#clf = GridSearchCV(MLPClassifier(), parameters)

#clf.fit(x_train,y_train)
#print(clf.score(x_train,y_train))
#print(clf.best_params_)

rfc = RandomForestClassifier(n_estimators = 18, random_state = 42)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
print("Female Count random")
print(np.count_nonzero(y_pred))
print("Male Count random")
print(len(y_pred)-np.count_nonzero(y_pred))
print("Random accuracy training",rfc.score(x_train,y_train))
print("Random accuracy",accuracy_score(y_pred,y_test))


clf = MultinomialNB(alpha=1.28101).fit(x_train,y_train)
prediction=clf.predict(x_test)
print("Female Count naive")
print(np.count_nonzero(prediction))
print("Male Count naive")
print(len(prediction)-np.count_nonzero(prediction))
print("multinobial accuracy training",clf.score(x_train,y_train))
print("multinobial accuracy: ",clf.score(x_test,y_test))

cf_svm = SVC(C=1.0,gamma=1.3).fit(x_train,y_train)
prediction_svm=cf_svm.predict(x_test)


print("Female Count svm")
print(np.count_nonzero(prediction_svm))
print("Male Count svm")
print(len(prediction_svm)-np.count_nonzero(prediction_svm))
print("svm accuracy training",cf_svm.score(x_train,y_train))
print("svm accuracy: ",cf_svm.score(x_test,y_test))



clf_mlp =MLPClassifier(hidden_layer_sizes=15,max_iter=2500).fit(x_train,y_train)
prediction_mlp=clf_mlp.predict(x_test)
print("mlp training: ",clf_mlp.score(x_train,y_train))
print("mlp accuracy: ",clf_mlp.score(x_test,y_test))
predictions_voting=[]
for i in range(len(prediction)):
    a=[prediction[i],y_pred[i],prediction_svm[i],prediction_mlp[i]]
    predictions_voting.append(max(set(a), key=a.count))

print(len(predictions_voting)-np.count_nonzero(predictions_voting))

print("manual voting Score -> ",accuracy_score(predictions_voting, y_test)*100)
x=["Random Forest","Naive Bayes","SVM","MLP","Voting"]
plt.plot(x,[0.9516943042537851,0.7359609068332933,0.9565809500921253, 0.954658335336057,0.89965],label="Training")
plt.plot(x,[0.6898429990387697,0.6754245434155719,0.6699775712912528,0.68727971803909,0.7036206344120475],label="Validation")

plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
