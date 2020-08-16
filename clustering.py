import pandas as pd
import numpy as np
import collections
import os
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

#change these setting as per your requirement
FILE_DIR = '/home/malar/k-means-clustering'
NEW_WIDTH = 400
NEW_HEIGHT = 300

def read_annot():
    '''read the bounding box annotation and other video details in the form of dataframe'''
    csv_file = os.path.join(FILE_DIR,'trainset.csv')
    return pd.read_csv(csv_file)


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        #the minimum value is the closest box to the cluster
        nearest_clusters = np.argmin(distances, axis=1)

        #program breaks if the last cluster equals nearest cluster means that 
        #they are so close to each other 
        if (last_clusters == nearest_clusters).all():
            plt.scatter(boxes[:, 0], boxes[:, 1], c=nearest_clusters, s=50, cmap='viridis')
            plt.scatter(clusters[:, 0], clusters[:, 1], c='black', s=200, alpha=0.5);
            #plt.plot([], [], ' ', label=clusters)
            #plt.legend()
            plt.show()
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters
 

def main():
    data = read_annot()
    data['b_w'] = NEW_WIDTH*data['boxw']/data['Imagew']
    data['b_h'] = NEW_HEIGHT*data['boxh']/data['Imageh']
    data['b_ar'] = data['b_w']/data['b_h']
    base_box = 16*16
    data['b_area_scale'] = (data['b_w']*data['b_h']/(base_box)).apply(np.sqrt)

    #plot the the available boxes to choose the base anchor size
    #sns.jointplot(x="b_w", y="b_h", data=data)
    #plt.show()

    X = data[data.columns[6:8]].to_numpy()
    #change this to 0 if you need to use eucledian distance as distance metric
    iou = 1
    # IOU based clustering
    if iou:
       cl = kmeans(X, 6)
       ar_iou = cl[:,0]/cl[:,1]
       scale_iou = cl[:,1]*np.sqrt(ar_iou)/16

    #Euclidean distance clustering
    else:
        from sklearn.cluster import KMeans
        K = KMeans(6, random_state=0)
        labels = K.fit(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels.labels_,
            s=50, cmap='viridis')  
        center = labels.cluster_centers_
        plt.scatter(center[:, 0], center[:, 1], c='black', s=200, alpha=0.5)
        plt.show()
        ar = center[:,0]/center[:,1]
        scale = center[:,1]*np.sqrt(ar)/16 # 16 is my base anchor size

if __name__ == "__main__":
   main()
 
