import os
import sys
import math
import numpy as np

def main():
    input_path = sys.argv[1]
    output_file = sys.argv[2]
    
    matrix_path = input_path

    f2 = open(output_file,'w')

    matrix_path_list = os.listdir(matrix_path)
    matrix_path_list.sort(key=lambda x:int(x[10:-4]))

    virus_count = -1
    for matrix_file_outer in matrix_path_list:
        f = open(matrix_path + '/' + matrix_file_outer)
        print(f.name)
        a = np.loadtxt(f) 
        s_a = a.shape
        print(a.shape)  
        virus_count = virus_count + 1 
            
        for matrix_file_inner in matrix_path_list:
            f1 = open(matrix_path + '/' + matrix_file_inner)
            print(f1.name)
            
            b = np.loadtxt(f1)
            s_b = b.shape
            print(b.shape)  
            row = min(s_a[0], s_b[0])
                    
            grid = np.zeros(shape=(568, 1))
            grid = np.sum(pow(a[:row,:] - b[:row,:],2),axis=1)
            print(grid.shape)       
        
            grid = grid.T
            np.savetxt(f2, grid, fmt='%f', delimiter=' ', newline=' ')
        
            f2.write('\n')
            
    f2.close()        
   
    
if __name__ == '__main__':
    main()