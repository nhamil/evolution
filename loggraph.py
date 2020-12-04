import sys 

import matplotlib.pyplot as plt 
import numpy as np 

plt.title("{} Fitness Score Over Time".format(sys.argv[1])) 
plt.xlabel("Generation") 
plt.ylabel("Log Fitness Score") 

def graph(filename: str): 
    f = open(filename, "r") 

    lines = f.readlines() 
    scores = [np.log(float(line.split(', ')[0])) for line in lines[1:]] 
    plt.plot(scores, label=lines[0][:-1].split(', ')[0]) 

    scores = [np.log(float(line.split(', ')[1])) for line in lines[1:]] 
    plt.plot(scores, label=lines[0][:-1].split(', ')[1]) 

for f in sys.argv[2:]: 
    graph(f) 

plt.legend() 
plt.show() 