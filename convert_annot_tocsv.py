import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os
import cv2
from cv2 import imread
from collections import defaultdict
import pandas as pd

#change these according to tyour requirements
ANNO_DIR = '/mnt/AI_RAID/UCF-Sports-Actions/ucfsports-anno'
TRAIN_LIST = '/mnt/AI_RAID/UCF-Sports-Actions/frames_list/train_list.txt'
FRAME_DIR = '/mnt/AI_RAID/UCF-Sports-Actions/Frames'
def read_annot():
        #keyList = ["image_width", "image_height","box_width", "box_height"] 
      
        # Using Dictionary comprehension 
        myDict = defaultdict(list) 
        count =0
        with open(TRAIN_LIST,'r') as file:
          for line in file:
              video_id = line.split()[0].split('/')[5]
              h_w_per_video = [imread(os.path.join(FRAME_DIR,video_id,i)).shape for i in sorted(os.listdir(os.path.join(FRAME_DIR,video_id)))]
              filename = [(os.path.join(video_id,i))for i in sorted(os.listdir(os.path.join(FRAME_DIR,video_id)))]
              assert len(filename) == len(h_w_per_video)
              for i in range(len(h_w_per_video)):
                  h,w,_ = h_w_per_video[i]
                  myDict["filename"].append(filename[i])
                  myDict["image_width"].append(w)
                  myDict["image_height"].append(h)
              with open((os.path.join(ANNO_DIR ,(video_id + '.txt'))),'r') as ann_file:
                  for ann in ann_file:
                     if int(ann.split()[0]) <= len(filename):
                       myDict["box_width"].append(ann.split()[3])
                       myDict["box_height"].append(ann.split()[4])
              assert len(myDict['filename']) == len(myDict['image_width']) ==len(myDict['box_width']) == len(myDict['image_height'])  ==len(myDict['box_height']) 
              count+=1
              
        print("completed")
        (pd.DataFrame.from_dict(data=myDict).to_csv('trainset.csv', header=False))
        print("Completed frames : {}".format(count))

def main():
    read_annot()

if __name__ == "__main__":
    main()
