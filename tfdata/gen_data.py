import os
import sys

f = open('data.txt','w')
labels = ['0','1']
image_lists=[]
print(os.listdir('data/train/'))
for label in labels:
    image_path='data/train/'+label
    for image_list in os.listdir(image_path):
        print(label+" "+image_path+"/"+image_list+"\n")
        f.write(label+" "+image_path+"/"+image_list+"\n")

f.close()

