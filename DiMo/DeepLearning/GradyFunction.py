import numpy as np
import csv

############################################
dataarray5lengtgh = []
with open('PatternsLevel5TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray5lengtgh.append(row)
###########################################
Dictionary = []
with open('Dictionary') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        Dictionary.append(row)


def divideLevel5(string):
    result = string
    temps = ''
    temp = ''
    x = string.split(' ')
    if (len(x[0]) == 2):
        temps = x[0]
        temp = temps[0] + ' ' + temps[1]
        result = temp + ' ' + x[1]

    if (len(x[2]) == 2):
        temps = x[2]
        temp = temps[0] + ' ' + temps[1]
        result = x[1] + ' ' + temp

    return result


####################4
def divideLevel4(string):
    result = string
    temps = ''
    temp = ''
    x = string.split(' ')
    if (len(x[0]) == 3):
        temps = x[0]
        temp = temps[0:2] + ' ' + temps[2]
        result = temp + ' ' + x[1]

    if (len(x[1]) == 3):
        temps = x[1]
        temp = temps[0:2] + ' ' + temps[2]
        result = x[0] + ' ' + temp
    if (len(x[2]) == 3):
        temps = x[2]
        temp = temps[0:2] + ' ' + temps[2]
        result = x[1] + ' ' + temp
    return result


####################3
def divideLevel3(string):
    result = string
    temps = ''
    temp = ''
    x = string.split(' ')
    if (len(x[0]) == 4):
        temps = x[0]
        temp = temps[0:3] + ' ' + temps[3]
        result = temp + ' ' + x[1]

    if (len(x[1]) == 4):
        temps = x[1]
        temp = temps[0:3] + ' ' + temps[3]
        result = x[0] + ' ' + temp
    if (len(x[2]) == 4):
        temps = x[2]
        temp = temps[0:3] + ' ' + temps[3]
        result = x[1] + ' ' + temp

    return result


#####################Default
def defaultDivided(string):
    dataarray5lengtgh1 = []
    np.array(dataarray5lengtgh)
    string
    s = string
    s1 = ''
    tl1 = 0
    temps = ''
    temp = ''
    max = 0
    dictionary = Dictionary
    tempstringDivided = np.array([])
    for l in range(len(s) - 4):
        tempsplit = s[l:l + 5]
        for da in dataarray5lengtgh:
            if tempsplit == da[0]:
                if float(da[5]) > max:
                    tl1 = l
                    break

    s1 = s[:tl1] + ' ' + s[tl1:tl1 + 5] + ' ' + s[tl1 + 5:]
    s1 = divideLevel5(s1)
    return s1


#######################################


def functionStringdivided(str):
    s = str
    s1 = str
    counter = 0
    tempstringDivided = np.array([])
    for l in range(len(s) - 4):
        tempsplit = s[l:l + 5]
        for d in Dictionary:
            if d == tempsplit:
                s1 = s[:l] + ' ' + s[l:l + 5] + ' ' + s[l + 5:]
                s1 = divideLevel5(s1)
                return s1
    counter = counter + 1
    #########################4
    counter = 0
    tempstringDivigmkded = np.array([])
    for l in range(len(s) - 3):
        tempsplit = s[l:l + 4]
        for d in Dictionary:
            if d == tempsplit:
                s1 = s[:l] + ' ' + s[l:l + 4] + ' ' + s[l + 4:]
                s1 = divideLevel4(s1)
                return s1
    counter = counter + 1

    #########################3
    for l in range(len(s) - 2):
        tempsplit = s[l:l + 3]
        for d in Dictionary:
            if d == tempsplit:
                s1 = s[:l] + ' ' + s[l:l + 3] + ' ' + s[l + 3:]
                s1 = divideLevel3(s1)
                return s1
    s1 = defaultDivided(s)
    return s1
