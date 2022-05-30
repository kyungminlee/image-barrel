import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=',')
    plt.plot(data[:,0], data[:,1], '.-')
    plt.show()

if __name__=='__main__':
    main()
