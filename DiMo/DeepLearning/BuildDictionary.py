## // Read Files
import csv
import numpy as np

##################### level5
dataarray5lengtgh = []
with open('PatternsLevel5TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray5lengtgh.append(row)
dataarray5lengtgh = np.array(dataarray5lengtgh)
print(dataarray5lengtgh)
##################### level4
dataarray4lengtgh = []
with open('PatternsLevel4TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray4lengtgh.append(row)
dataarray4lengtgh = np.array(dataarray4lengtgh)
print(dataarray4lengtgh)
############################ level3
dataarray3lengtgh = []
with open('PatternsLevel3TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray3lengtgh.append(row)
dataarray3lengtgh = np.array(dataarray3lengtgh)
print(dataarray3lengtgh)

############################ level2
dataarray2lengtgh = []
with open('PatternsLevel2TSD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        dataarray2lengtgh.append(row)
dataarray2lengtgh = np.array(dataarray2lengtgh)
print(dataarray2lengtgh)
# //  Create Dictionary
dictionary = []
for l in dataarray5lengtgh:
    temp = float(l[5])
    if temp > 0.5:
        dictionary.append(l[0])

for l in dataarray4lengtgh:
    temp = float(l[5])
    if temp > 0.5:
        dictionary.append(l[0])

for l in dataarray3lengtgh:
    temp = float(l[5])
    if temp > 0.5:
        dictionary.append(l[0])

for l in dataarray2lengtgh:
    temp = float(l[5])
    if temp > 0.5:
        dictionary.append(l[0])

# // Save Dictionary to file
with open('Dictionary', 'w', newline='') as csv_file:
    for p in dictionary:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(p)

#####################################################Functions###########################
