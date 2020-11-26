import sys 

import matplotlib.pyplot as plt 

plt.title("{} Fitness Score Over Time".format(sys.argv[1])) 
plt.xlabel("Generation") 
plt.ylabel("Fitness Score") 

def graph(filename: str): 
    f = open(filename, "r") 

    lines = f.readlines() 
    scores = [float(line) for line in lines[1:]] 
    plt.plot(scores, label=lines[0][:-1]) 

for f in sys.argv[2:]: 
    graph(f) 

plt.legend() 
plt.show() 