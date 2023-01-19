"""
    Investigates HV offset values and creates plots
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv("./HV_offset_57.csv")
    print(data.shape)
    l = len(data["constant"])
    values,errors = np.zeros(l),np.zeros(l)
    for i in range(l):
        values = (data["measured current"]+data["measured current 2"])/2.0
        errors = 5.0
    plt.figure()
    plt.grid()
    plt.errorbar(data["constant"],values,errors)
    plt.xlim(-600,600),plt.ylim(-200,200)
    plt.savefig("HV_Offset_57.png")

if __name__ == '__main__':
    main()