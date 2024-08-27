#Calculate cosine similarity between two embedding vectors of protein sequences.
import cv2
import cupy as np
import math
import time
import os
import sys

def main():
    input_path = sys.argv[1]
    output_file = sys.argv[2]

    matrix_path = input_path

    matrix_path_list = os.listdir(matrix_path)
    matrix_path_list.sort(key=lambda x:int(x[10:-4]))

    f1 = open(output_file,'w')

    for matrix_file_outer in matrix_path_list:
        f = open(matrix_path + '/' + matrix_file_outer)
        a = np.loadtxt(f) 
        s_a = a.shape 
        
        for matrix_file_inner in matrix_path_list:
            f1 = open(matrix_path + '/' + matrix_file_inner)
            b = np.loadtxt(f1)
            s_b = b.shape
            row = min(s_a[0], s_b[0])
                    
            grid = np.zeros(shape=(row, 1))

            for i in range(row):
                vector_1 = a[i, :]
                vector_2 = b[i, :]
                cosine = (float(np.dot(vector_1, vector_2)) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
                normalized_dist = (1-cosine)/ 2.0
                grid[i] = normalized_dist
            grid = grid.T
            np.savetxt(f1, grid, fmt='%f', delimiter=',', newline=' ')
        
            f1.write('\n')
    f1.close()        


if __name__ == '__main__':
    main()
