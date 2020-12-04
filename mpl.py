import matplotlib.pyplot as plt 
import numpy as np 

if __name__ == "__main__":
    plt.ion() 
    y = [0.0] 

    for i in range(10): 
        y.append(y[-1] + np.random.randn()) 
        # ax.plot(y) 
        plt.cla() 
        plt.plot(y) 

        plt.pause(0.001) 

    plt.pause(-1) 