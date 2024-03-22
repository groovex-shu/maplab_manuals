import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import fnmatch

def find_files(dir_path, pattern='*.txt'):
    matches = []
    for root, dirs, files in os.walk(dir_path):
        for filename in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, filename))
    
    return matches  



if __name__ == "__main__":
    # read csv
    name = '20240308_101434_kfh5'
    root = f'/data/localization/maps/{name}'


    dist_edges = find_files(root, pattern='*distance_edges.csv')
    df = pd.read_csv(dist_edges[0])
    df = df.iloc[:, -3:]
    

    edges_np = df.to_numpy()
    print(edges_np.shape)
    

    csvs = find_files(root, pattern='*vertices.csv')
    # print(csvs)
    
    csv_np = []
    for csv in csvs:
        csv_np.append(np.loadtxt(csv, delimiter=',', skiprows=1))
        print(csv_np[-1].shape) 

    # craete plot
    fig, ax = plt.subplots(1, len(csv_np))


    
    dist_list = []
    for i, csv in enumerate(csv_np):
        vertices = csv[:, 2:5]
        dist = []
        pre_vertice = None
        for vectice in vertices:
            if pre_vertice is not None:
                dist.append(np.linalg.norm(vectice - pre_vertice))
            
            pre_vertice = vectice

        dist_list.append(np.array(dist))
        print(dist_list[-1].shape)

        ax[i].plot(dist_list[-1])
    
    plt.show()

    plt.scatter(np.arange(len(dist_list[-1])), dist_list[-1], c='b')
    plt.scatter(edges_np[:, 1], edges_np[:, 2], c='r')

    vertice0 = np.array(dist_list[-1])
    vertice0 = vertice0[edges_np[:, 1].astype(int)]
    assert vertice0.shape == edges_np[:, 2].shape, 'index select error'

    diff = vertice0 - edges_np[:, 2]
    
    plt.plot(edges_np[:, 1], diff, color='g', label='Diff', linestyle='-')
    for x_val in edges_np[:, 1]:
        plt.axvline(x=x_val, color='gray', linestyle='-')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Difference between scatter A and B')


    plt.show()

    

    


