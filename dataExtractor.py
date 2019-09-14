#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import requests
import cv2 as cv
from PIL import Image
import os
import sys
import glob
import pandas as pd
import tensorflow as tf


# In[2]:


#reads the json file

filename = 'data_images.json'
jsonData =[]
count =0
with open(filename, "r") as jsonfile:
    for line in jsonfile:
        count +=1
        jsonData.append(json.loads(str(line.strip())))
        
print("number of rows:",count)


# In[5]:


#Download images

path = './images/image_'
count = 1
for image_data in jsonData:
    url = image_data['content']
    r = requests.get(url)
    ext = url.split('.')[-1]
    with open(path+str(count)+'.'+ext,'wb') as f: 
        f.write(r.content)
    count +=1


# In[ ]:





# In[ ]:


from_range = 1
to_range = 409
filename = 'trainval.txt'
count = 1
with open(filename, 'w') as file:
    for count in range(1,409):
        file.write("image_"+str(count)+"\n")
        count = count+1


# In[70]:


#create the dataframe in required format

#For trainSet use from_range= 1 to_range=300 filename = train_label.csv
#For testset use from_range = 301 to_range=409 filename = test_label.csv

from_range = 1
to_range = 300
filename = 'train_label.csv'

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
output_df_list = []
path = 'image_'
count = 1
for i in range(from_range,to_range):
    #print(i)
    ext = jsonData[i-1]['content'].split(".")[-1]
    for t in jsonData[i-1]['annotation']:
        label = t['label']
        points = t['points']
        imageWidth = int(t['imageWidth'])
        imageHeight = int(t['imageHeight'])
        x1 = int(float(points[0]['x'])*imageWidth)
        y1 = int(float(points[0]['y'])*imageHeight)
        x2 = int(float(points[1]['x'])*imageWidth)
        y2= int(float(points[1]['y'])*imageHeight)
        xmin = min(x1,x2)
        ymin = min(y1,y2)
        xmax = max(x1,x2)
        ymax = max(y1,y2)
        width = xmax-xmin
        height = ymax-ymin
        #print(width,height)
        if(width >0 and height > 0):
            image_list = [path+str(i)+"."+ext, width,height,'face',xmin,ymin,xmax,ymax]
            output_df_list.append(image_list)
            
xml_df = pd.DataFrame(output_df_list, columns=column_name)
xml_df.to_csv(filename, index=None)


# In[82]:


#create the dataframe in required format

#For trainSet use from_range= 1 to_range=300 filename = train_label.csv
#For testset use from_range = 301 to_range=409 filename = test_label.csv

from_range = 301
to_range = 409
filename = 'test_label.csv'

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
output_df_list = []
path = 'image_'
count = 1
for i in range(from_range,to_range):
    #print(i)
    ext = jsonData[i-1]['content'].split(".")[-1]
    for t in jsonData[i-1]['annotation']:
        label = t['label']
        points = t['points']
        imageWidth = int(t['imageWidth'])
        imageHeight = int(t['imageHeight'])
        x1 = int(float(points[0]['x'])*imageWidth)
        y1 = int(float(points[0]['y'])*imageHeight)
        x2 = int(float(points[1]['x'])*imageWidth)
        y2= int(float(points[1]['y'])*imageHeight)
        xmin = min(x1,x2)
        ymin = min(y1,y2)
        xmax = max(x1,x2)
        ymax = max(y1,y2)
        width = xmax-xmin
        height = ymax-ymin
        #print(width,height)
        if(width >0 and height > 0):
            image_list = [path+str(i)+"."+ext, width,height,'face',xmin,ymin,xmax,ymax]
            output_df_list.append(image_list)
            
xml_df = pd.DataFrame(output_df_list, columns=column_name)
xml_df.to_csv(filename, index=None)


# In[64]:


#Dummy

#xml_df = pd.DataFrame(output_df_list, columns=column_name)


# In[65]:


#dummy
#xml_df.to_csv('image_labels.csv', index=None)
#print('Successfully converted xml to csv.')


# In[ ]:





# In[ ]:




