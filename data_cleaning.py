
from PIL import Image
import glob
import shutil



fileExtensions = ["jpg", "png"]
originalFilenamesTemp = []
pathO = r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\originals\*.'
#pathO = r'D:\originals\*.'

for extension in fileExtensions:
    templist = [img for img in glob.glob(pathO+extension)]
    originalFilenamesTemp.extend(templist)


#------ Create Dictionary of Data Examples ------


data = {}
AllFilenames =[]

AllFilenames += originalFilenamesTemp


filenames = []

for file in originalFilenamesTemp:
    split = file.split('\\')
    name = split[-1][:-4]
    filenames.append(name)
    data[name] = [file]


pathP = r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\photoshops\{}\*.{}'
#pathP = r'D:\photoshops\*.'

for x in filenames:
    photoshoppedFilenamesTemp = []
    for extension in fileExtensions:
        string = pathP.format(x,extension)
        templist = [img for img in glob.glob(string)]
        photoshoppedFilenamesTemp.extend(templist)
    data[x] += [photoshoppedFilenamesTemp]
    AllFilenames += photoshoppedFilenamesTemp

#print(data) 


#--------------- Histogram of Dimension ----------
from PIL import Image
import os.path

dimensions = []
pixelSize = []
for y in AllFilenames:
    filename = os.path.join(y)
    img = Image.open(filename)
    dimensions.append(img.size)
    pixelSize.append(img.size[0]*img.size[1])

#print(dimensions,'\n\n\n',pixelSize)

import matplotlib.pyplot as plt
import numpy as np

plt.hist(pixelSize, normed=True, bins=30)
plt.xlabel('Pixel Dimension')
plt.ylabel('Frequency')

#------------- Slipting the Data Into Lists------------------
trainSet1 = []
trainSet0 = []
valSet1 = []
valSet0 = []
testSet1 = []
testSet0 = []

i = 0


for z in data.keys():
    value = data[z]
    if len(value[1]) > 0:
        if i == 0 or i ==1 or i == 2:
            trainSet1.append(value[1][0])
        if i == 3 or i ==4 or i == 5:
            trainSet0.append(value[0])
        if i == 6:
            valSet1.append(value[1][0])
        if i == 7:
            valSet0.append(value[0])
        if i == 8:
            testSet1.append(value[1][0])
        if i == 9:
            testSet0.append(value[0])
        
        i += 1
    
    if (i % 10) == 0:
        i = 0
    

#--------------Split data on PC--------------
for trData1 in trainSet1:
    shutil.move(trData1,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\trainData\1')
for trData0 in trainSet0:
    shutil.move(trData0,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\trainData\0')
    
for valData1 in valSet1:
    shutil.move(valData1,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\valData\1')
for valData0 in valSet0:
    shutil.move(valData0,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\valData\0')
    
for teData1 in testSet1:
    shutil.move(teData1,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\testData\1')
for teData0 in testSet0:
    shutil.move(teData0,r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\testData\0')



##--------------Split data on Hard Drive--------------
#for trData1 in trainSet1:
#    shutil.move(trData1,r'D:\trainData\1')
#for trData0 in trainSet0:
#    shutil.move(trData0,r'D:\trainData\0')
#    
#for valData1 in valSet1:
#    shutil.move(valData1,r'D:\valData\1')
#for valData0 in valSet0:
#    shutil.move(valData0,r'D:\valData\0')
#    
#for teData1 in testSet1:
#    shutil.move(teData1,r'D:\testData\1')
#for teData0 in testSet0:
#    shutil.move(teData0,r'D:\testData\0')




