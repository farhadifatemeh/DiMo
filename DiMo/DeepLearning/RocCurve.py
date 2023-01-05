import sklearn
from gensim.test.utils import get_tmpfile
from gensim.models import word2vec, Word2Vec
from keras.layers import Dense
from keras.layers import  SimpleRNN, LSTM, GRU
from keras.models import Sequential
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
from EvaluateMetrics import recall_m, precision_m, f1_m

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
############## Test
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
# ///////// NLP
# // CNN Model
modelCNN = Sequential()
modelCNN.add(Dense(302, activation='relu'))
modelCNN.add(Dense(150, activation='relu'))
modelCNN.add(Dense(75, activation='relu'))
modelCNN.add(Dense(1, activation='sigmoid'))
modelCNN.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
history = modelCNN.fit(x_train, ytrain, epochs=5, batch_size=500, validation_split=0.2)
modelCNN.summary()

x_trainRNN = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_testRNN = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# // RNN Model
modelRNN = Sequential()
modelRNN.add(SimpleRNN(100, input_shape=(302, 1), return_sequences=True))
modelRNN.add(SimpleRNN(100, input_shape=(302, 1)))
modelRNN.add(Dense(1, activation='sigmoid'))
modelRNN.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
history = modelRNN.fit(x_trainRNN, ytrain, epochs=5, batch_size=500, validation_split=0.2)
modelRNN.summary()
# // LSTM Model
modelLSTM = Sequential()
modelLSTM.add(LSTM(100, input_shape=(302, 1)))
modelLSTM.add(Dense(1, activation='sigmoid'))
modelLSTM.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
history = modelLSTM.fit(x_trainRNN, ytrain, epochs=5, batch_size=158, validation_split=0.2)
modelLSTM.summary()
#  // GRU Model
modelGRU = Sequential()
modelGRU.add(GRU(100, input_shape=(302, 1)))
modelGRU.add(Dense(1, activation='sigmoid'))
modelGRU.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
history = modelGRU.fit(x_trainRNN, ytrain, epochs=5, batch_size=500, validation_split=0.2)
modelGRU.summary()
# // Naive Baysian Model
kfold = KFold(shuffle=True)
modelNB = GaussianNB()
resultNB = cross_val_score(modelNB, x_train, ytrain, cv=kfold)
modelNB.fit(x_train, ytrain)
# // Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(x_train, ytrain)
rray = decision_tree.predict(x_test)

# // Evaluate Metrics
temp = 0
print(len(rray))
for i in range(0, len(rray)):
    if abs(rray[i] - ytest[i] > 0):
        temp = temp + 1
print("Count", temp)
Acc_DecisionTree = sklearn.metrics.accuracy_score(ytest, rray, normalize=True, sample_weight=None)
print("Acc_DecisionTree: ", Acc_DecisionTree)
print("precision_DecisionTree: ", sklearn.metrics.precision_score(ytest, rray, average='weighted'))
print("recall_DecisionTree: ", sklearn.metrics.recall_score(ytest, rray, average='weighted'))
print("pfscorerecision_recall_DecisionTree: ",
      sklearn.metrics.precision_recall_fscore_support(ytest, rray, average='weighted'))
print("\n")
########################################################
print("accuracy_scoreNB: ", accuracy_score(ytest, modelNB.predict(x_test)))
print("precision_scoreNB: ", precision_score(ytest, modelNB.predict(x_test), average='weighted'))
print("recall_scoreNB: ", recall_score(ytest, modelNB.predict(x_test), average='weighted'))
print("precision_recall_fscore_scoreNB: ",
      sklearn.metrics.precision_recall_fscore_support(ytest, modelNB.predict(x_test), average='weighted'))
print("\n")
########################
loss, accuracy, f1_score, precision, recall = modelCNN.evaluate(x_test, ytest)
print("lCNN: ", loss, " accCNN: ", accuracy, " f1CNN: ", f1_score, " preCNN: ", precision, " recCNN: ", recall, "\n")
##########################

loss, accuracy, f1_score, precision, recall = modelLSTM.evaluate(x_testRNN, ytest)
print("lLSTM: ", loss, " accLSTM: ", accuracy, " f1LSTM: ", f1_score, " preLSTM: ", precision, " recLSTM: ", recall,
      "\n")
###################################
loss, accuracy, f1_score, precision, recall = modelGRU.evaluate(x_testRNN, ytest)
print("lGRU: ", loss, " accGRU: ", accuracy, " f1GRU: ", f1_score, " preGRU: ", precision, " recGRU: ", recall, "\n")
##############################
loss, accuracy, f1_score, precision, recall = modelRNN.evaluate(x_testRNN, ytest)
print("lRNN: ", loss, " accRNN: ", accuracy, " f1RNN: ", f1_score, " preRNN: ", precision, " recRNN: ", recall, "\n")
###############################
# // Roc Curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
ax = plt.axes()
ax.set_facecolor("#D6D8DA")
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
fprcnn, tprcnn, thresholdscnn = roc_curve(ytest, modelCNN.predict(x_test))
auc = roc_auc_score(ytest, modelCNN.predict(x_test))
plt.plot(fprcnn, tprcnn, label='%s ROC (area = %0.2f)' % ('DiMo-C', auc))

#############################################
fprrnn, tprrnn, thresholdsrnn = roc_curve(ytest, modelGRU.predict(x_test))
print(fprrnn, ".........................")
auc = roc_auc_score(ytest, modelLSTM.predict(x_test))
# Now, plot the computed values
plt.plot(fprrnn, tprrnn, label='%s ROC (area = %0.2f)' % ('DiMo-R', auc))
# Custom settings for the plot
###################################################
fprlstm, tprlstm, thresholdslstm = roc_curve(ytest, modelLSTM.predict(x_test))
print(fprlstm, ".........................")
auc = roc_auc_score(ytest, modelLSTM.predict(x_test))
plt.plot(fprlstm, tprlstm, label='%s ROC (area = %0.2f)' % ('DiMo-L', auc))

#############################################################
fprgru, tprgru, thresholdsgru = roc_curve(ytest, modelGRU.predict(x_test))
print(fprlstm, ".........................")
auc = roc_auc_score(ytest, modelGRU.predict(x_test))
plt.plot(fprgru, tprgru, label='%s ROC (area = %0.2f)' % ('DiMo-G', auc))

#############################################################
fprdt, tprdt, thresholdsdt = roc_curve(ytest, decision_tree.predict(x_test))
print(fprlstm, ".........................")
auc = roc_auc_score(ytest, decision_tree.predict(x_test))
plt.plot(fprdt, tprdt, label='%s ROC (area = %0.2f)' % ('DiMo-D', auc))
#############################################################
fprnb, tprnb, thresholdsnb = roc_curve(ytest, modelNB.predict(x_test))
print(fprlstm, ".........................")
auc = roc_auc_score(ytest, modelNB.predict(x_test))
plt.plot(fprnb, tprnb, label='%s ROC (area = %0.2f)' % ('DiMo-NB', auc))
# // Draw Plot

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
