# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:46:32 2024

@author: Noopa Jagadeesh
"""

### Reading multiple images from a folder
#The glob module finds all the path names, 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 


#import the library opencv
import cv2
import glob
file_list = glob.glob('COSC/images/test/*.*') #Rerurns a list of file names
print(file_list)  #Prints the list containing file names

#------------------------------------------------------------------------------
#Now let us load each file at a time...
my_list=[]  #Empty list to store images from the folder.
path = "COSC/images/test/*.png*"
for file in glob.glob(path):   #Iterate through each file in the list using for
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    my_list.append(a)  #Create a list of images (not just file names but full images)
    
#View images from the stored list
from matplotlib import pyplot as plt
plt.imshow(my_list[2])  #View the 3rd image in the list.