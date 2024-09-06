#Calculate antigenic distance for nowcast task


import math
import numpy as np

def main():
  f_before = sys.argv[1]
  f_after = sys.argv[2]

  x1 = []
  x2 = []
  y1 = []
  y2 = []
  
  for virus in f_before:
      values = virus.split(' ')
      x1.append(float(values[-2]))
      y1.append(float(values[-1]))
  
  for virus in f_after:
      values = virus.split(' ')
      x2.append(float(values[-2]))
      y2.append(float(values[-1]))
  
  print(len(x1))
  print(len(y1))
  
  x1, x2 = np.meshgrid(x1,x2)
  y1, y2 = np.meshgrid(y1,y2)
  
  dis = np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
  
  f_dis = open('nowcast-season-2005-fs/antigenicDis/h3-dis-nowcast-vaccine-2006-ss.csv','w')
  
  np.savetxt(f_dis, dis, delimiter=',', fmt='%.04f') 
  
  print('Maximum of Y is ' + str(np.max(dis)) + '. Minimum of Y is ' + str(np.min(dis)))

if __name__ == '__main__':
    main()
