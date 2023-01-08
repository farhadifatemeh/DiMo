import pymysql
from numpy import *
import numpy as np
import csv
from Functions import absSecond, fiveSecond, occurrences, listToString
DataNoneMotifs = []
DataMotifs = []
motifstrs = [""]
Nonemotifstrs = [""]
resultPatterns1 = []
resultPatterns2 = []


def binomial(val):
    x = abs(int(val))
    # b = -0.161 * (x * x) + (8.055 * x)+0.06039
    # b = -0.1508 * (x * x) + (7.784 * x) + 0.02657
    b = -0.3265 * (x * x) + (11.58 * x) + 1.387
    return b


#    /////// Read Dataset, input:Paths
Motifs = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\FunctionOnFrfequency000523\Datasets\Train\TSD\Motifs.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        Motifs.append(row)
Sequences = []
with open(
        'E:\Courses991028\PythonProject\pythonProject14000506\FunctionOnFrfequency000523\Datasets\Train\TSD\Sequences.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        Sequences.append(row)

# ////////////////////////////////////////////////////

for x in range(0, 2199):
    DataMotifs.append(Motifs[x])
print(len(DataMotifs))

for x in range(0, 2199):
    temp = listToString(Sequences[x])
    DataNoneMotifs.append(temp[-7:])

######################################################################
##  // Building K-mers
resultPatterns3 = []
PatternsArray2 = []
tempArray = []
resultPatterns2 = [["A", 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], ["C", 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                   ["G", 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                   ["U", 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]
j = 0
Tempchar = ''
for respat in resultPatterns2:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    resultPatterns3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    resultPatterns3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    resultPatterns3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    resultPatterns3.append(tempArray)

# ////////////////////////////////////////////////////
###################LEVEL3##########################################################
# resultPatterns2 = []
PatternsArray3 = []
tempArray = []

j = 0
Tempchar = ''
for respat in resultPatterns3:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray3.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray3.append(tempArray)
###################LEVEL44444444444444444444444444$$$$##########################################################
# resultPatterns2 = []
PatternsArray4 = []
tempArray = []

j = 0
Tempchar = ''
for respat in PatternsArray3:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray4.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray4.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray4.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray4.append(tempArray)

# ////////////////////////////////////////////////////
###################LEVEL55555555555555555555$$$$##########################################################
# resultPatterns2 = []
PatternsArray5 = []
tempArray = []

j = 0
Tempchar = ''
for respat in PatternsArray4:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray5.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray5.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray5.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray5.append(tempArray)

# ////////////////////////////////////////////////////
###################LEVEL66666666666666666666666666666##########################################################
# resultPatterns2 = []
PatternsArray6 = []
tempArray = []

j = 0
Tempchar = ''
for respat in PatternsArray5:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray6.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray6.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray6.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray6.append(tempArray)

# ////////////////////////////////////////////////////
###################LEVEL7777777777777777777777777777##########################################################
# resultPatterns2 = []
PatternsArray7 = []
tempArray = []

j = 0
Tempchar = ''
for respat in PatternsArray6:
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'A'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray7.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'C'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray7.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'G'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray7.append(tempArray)
    tempArray = list(respat)
    tempArray[0] = respat[0] + 'U'
    tempArray[1] = 0.00
    tempArray[2] = 0.00
    tempArray[3] = 0.00
    tempArray[4] = 0.00
    tempArray[5] = 0.00
    tempArray[6] = 0.00
    PatternsArray7.append(tempArray)

# ////////////////////////////////////////////////////
countofTwolongMotif = 0
TempCounter = 0
i = 0
tempc = 0
lenghtemp = []
for tempp in DataMotifs:
    for p5 in PatternsArray7:
        if p5[0] in tempp:
            tempc = tempc + occurrences(listToString(tempp), listToString(p5[0]))

for p in PatternsArray7:
    for s in DataMotifs:
        TempCounter = TempCounter + occurrences(listToString(s), listToString(p[0]))
        countofTwolongMotif = countofTwolongMotif + occurrences(listToString(s), listToString(p[0]))
    PatternsArray7[i][1] = TempCounter
    TempCounter = 0
    i = i + 1
# //////////////////////////
TempCounterNoneMotif = 0
inm = 0
countofTwolong = 0
lenghtempn = 0
for tempp in DataNoneMotifs:
    lenghtempn = len(tempp)
for p in PatternsArray7:
    for s in DataNoneMotifs:
        TempCounterNoneMotif = TempCounterNoneMotif + occurrences(listToString(s), listToString(p[0]))
        countofTwolong = countofTwolong + occurrences(listToString(s), listToString(p[0]))
    PatternsArray7[inm][2] = TempCounterNoneMotif
    TempCounterNoneMotif = 0
    inm = inm + 1


#####################################################################################
mines=[]
for p in PatternsArray7:
    mines.append(int(p[1]) - int(p[2]))
for p in PatternsArray7:
    p[3] = format(float((int(p[1]) / 2199) * 100), '.3f')
    p[4] = format(float((int(p[2]) / 2199) * 100), '.3f')
    if int(p[1]) == 0 and int(p[2]) != 0:
       p[5] = 0
       p[6] = format(float(binomial(abs(int(p[1]) - int(p[2])))), '.3f')
    elif int(p[1]) != 0 and int(p[2]) == 0:
        p[5] = format(float(binomial(abs(int(p[1]) - int(p[2])))), '.3f')
        p[6] = 0
    elif int(p[1]) != 0 and int(p[2]) != 0:
        abs1=(abs(int(p[1]) - int(p[2]))/np.max(mines))
        plus=int(p[1]) + int(p[2])
        p[5] =format((float(int(p[1])/plus)*abs1)*100, '.3f')
        p[6]= format((float(int(p[2])/plus)*abs1)*100, '.3f')
    else:
        p[5] = 0
        p[6] = 0
################################################################################
correctionlevel7 = []
for p in PatternsArray7:

    if float(p[5]) > 1.5 or float(p[5]) < 5 or abs(int(p[1])-int(p[2])) > 10:

        correctionlevel7.append(p)
#######################################################################
correctionlevel7.sort(key=fiveSecond,reverse=True)
correctionlevel7.sort(key=absSecond,reverse=True)

# ///////////////////////////////////////////////////////////////////////////////////////////////
print("ggggg", tempc)
print(TempCounter)
print(PatternsArray4)
print(PatternsArray5)
print(countofTwolongMotif)
print(countofTwolong)
# ///////////////////////////////////////////////////

with open('PatternsLevel7TSD', 'w') as csv_file:
    for p in PatternsArray7:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

with open('PatternsLevel7TSD-sorted', 'w') as csv_file:
    for p in correctionlevel7:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)
