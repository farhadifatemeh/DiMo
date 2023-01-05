from sklearn.utils import shuffle
import csv
from ListtoString import listToString

# // Read train and test data
DataMotifs = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\pythonProject1\DeepLearning-sorted\DiMo\Datasets\Train\TSD\Motifs.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        DataMotifs.append(row)

DataNoneMotifs = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\pythonProject1\DeepLearning-sorted\DiMo\Datasets\Train'
        '\TSD\Sequences.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        DataNoneMotifs.append(row)

DatanonMotifTest = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\pythonProject1\DeepLearning-sorted\DiMo\Datasets\Test\TSD\Sequences.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        DatanonMotifTest.append(row)

DataMotifTest = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\pythonProject1\DeepLearning-sorted\DiMo\Datasets\Test\TSD\Motifs.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        DataMotifTest.append(row)

# // Read  data weights
import csv

dataarray = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\pythonProject1\DeepLearning-sorted\PatternsLevel7TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray.append(row)
outputData = []
for d in dataarray:
    t = []
    t.append(d[0])
    t.append(d[5])
    t.append(d[6])
    outputData.append(t)

##################################################
randomStringArray = []
randomStringArrayTest = []
length = 22
for blah in DataNoneMotifs:
    t = listToString(blah)
    randomStringArray.append(t[-7:])

for blah in DatanonMotifTest:
    t = listToString(blah)
    randomStringArrayTest.append(t[-7:])
# //
DataMotifsTest = []
DataMotifsTrain = []
ytest = []
ytrain = []
y = 1
n = 0
for x in DataMotifs:
    DataMotifsTrain.append(x)
    ytrain.append(y)

for x in DataMotifTest:
    DataMotifsTest.append(x)
    ytest.append(y)
print(len(DataMotifsTrain))

for x in randomStringArray:
    DataMotifsTrain.append(x)
    ytrain.append(n)
for x in randomStringArrayTest:
    DataMotifsTest.append(x)
    ytest.append(n)

# // Shuffle
DataMotifsTrain, ytrain = shuffle(DataMotifsTrain, ytrain, random_state=0)
DataMotifsTest, ytest = shuffle(DataMotifsTest, ytest, random_state=0)
# // Print result
print(len(DataMotifsTrain), ":", DataMotifsTrain)
print(len(ytrain), ":", ytrain)
print(len(DataMotifsTest), ":", DataMotifsTest)
print(len(ytest), ":", ytest)

# // Get weights
ci1 = 0
outputDataMotif = []
temp = []
for d in DataMotifsTrain:
    for w in outputData:
        if d == w[0]:
            temp = []
            temp.append(int(float(w[1]) * 100))
            temp.append(int(float(w[2]) * 100))
            break
    outputDataMotif.append(temp)
    ci1 = ci1 + 1
outputDataMotifTEST = []
ci1 = 0
for d in DataMotifsTest:
    for w in outputData:
        if d == w[0]:
            temp = []
            temp.append(int(float(w[1]) * 100))
            temp.append(int(float(w[2]) * 100))
            break
    outputDataMotifTEST.append(temp)
    ci1 = ci1 + 1
# // Save result to files
import csv

with open('OutputWeight_train4.csv', 'w', newline='') as csv_file:
    for p in outputDataMotif:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

with open('outputWeight_test4.csv', 'w', newline='') as csv_file:
    for p in outputDataMotifTEST:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

with open('Outputx_train4.csv', 'w', newline='') as csv_file:
    for p in DataMotifsTrain:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

with open('outputx_test4.csv', 'w', newline='') as csv_file:
    for p in DataMotifsTest:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

with open('Outputy_train4.csv', 'w', newline='') as csv_file:
    for p in ytrain:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(str(p))

with open('outputy_test4.csv', 'w', newline='') as csv_file:
    for p in ytest:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(str(p))
