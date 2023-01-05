import sklearn
from gensim.test.utils import get_tmpfile
from gensim.models import word2vec, Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import numpy as np
# // Call functions
from GradyFunction import functionStringdivided
from ListtoString import listToString

DataNoneMotifs = []
DataMotifsTrain = []
resultPatterns1 = []
resultPatterns2 = []
ytrain = []
# // Read train and test data
X_train = []
with open('X_trainTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        X_train.append(row)

Y_train = []
with open('Y_trainTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        Y_train.append(float(listToString(row)))

X_test = []
with open('X_testTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        X_test.append(row)

Y_test = []
with open('Y_testTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        Y_test.append(float(listToString(row)))

TestWeights = []
with open('TestWeightTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        TestWeights.append(row)

TrainWeights = []
with open('TrainWeightsTSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        TrainWeights.append(row)

# // Break Patterns
strresult = []
for stri in X_train:
    strresult.append(functionStringdivided(listToString(stri)))
########################### TEST
strresultTest = []
for stri in X_test:
    strresultTest.append(functionStringdivided(listToString(stri)))
# // Tokenised
tokenized_seq = [sentence.split() for sentence in strresult]
tokenized_seqTest = [sentence.split() for sentence in strresultTest]
# print("token\n",tokenized_seq)
dictionaryWord = []
for t in tokenized_seq:
    dictionaryWord.append(t)
for t in tokenized_seqTest:
    dictionaryWord.append(t)
path = get_tmpfile("word2vec.model")
model = word2vec.Word2Vec(dictionaryWord, min_count=1)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train(["hello", "World"], total_examples=1, epochs=1)
##################################
from gensim.models import KeyedVectors

# Store just the words + their trained embeddings.
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
word = word_vectors.load("word2vec.wordvectors")
vector = []
i = 0
for v in range(0, len(wv)):
    vector.append(wv[i])
    i = i + 1
c = 'A'

# // Get Feature Vector
tempword = []
svector = []
j = 0
for str in tokenized_seq:
    for s in str:
        tempword.append(word[s])
    svector.append(tempword)
    tempword = []
    j = j + 1

tempword = []
svectorTest = []
j = 0
for str in tokenized_seqTest:
    for s in str:
        tempword.append(word[s])
    svectorTest.append(tempword)
    tempword = []
    j = j + 1
####################
svector = np.array(svector)
Al = []
for p in svector:
    A = np.asarray(p).reshape(-1)
    Al.append(A)
Al = np.array(Al)
print(Al.shape)
############ TEST
AlTest = []
for p in svectorTest:
    A = np.asarray(p).reshape(-1)
    AlTest.append(A)
AlTest = np.array(AlTest)
print(AlTest.shape)
###################
temp1 = []
x_train = []
j = 0
for x in TrainWeights:
    temp1 = np.hstack((float(x[0]), float(x[1])))
    temp1 = np.hstack((Al[j], temp1))
    x_train.append(temp1)
    temp1 = []
    j = j + 1
x_train = np.array(x_train)
print(x_train.shape)
ytrain = np.array(ytrain)
##############Test

temp2 = []
x_test = []
j = 0
for x in TestWeights:
    temp2 = np.hstack((float(x[0]), float(x[1])))
    temp2 = np.hstack((AlTest[j], temp2))
    x_test.append(temp2)
    temp2 = []
    j = j + 1
x_test = np.array(x_test)
print(x_test.shape)
ytrain = np.array(Y_train)
ytest = np.array(Y_test)
# ////// Machine Learning
# // Naive Baysian Model
kfold = KFold(shuffle=True)
modelNB = GaussianNB()
resultNB = cross_val_score(modelNB, x_train, ytrain, cv=kfold)
modelNB.fit(x_train, ytrain)

# // Evaluate Metrics
print("accuracy_scoreNB: ", accuracy_score(ytest, modelNB.predict(x_test)))
print("precision_scoreNB: ", precision_score(ytest, modelNB.predict(x_test), average='weighted'))
print("recall_scoreNB: ", recall_score(ytest, modelNB.predict(x_test), average='weighted'))
print("precision_recall_fscore_scoreNB: ",
      sklearn.metrics.precision_recall_fscore_support(ytest, modelNB.predict(x_test), average='weighted'))
