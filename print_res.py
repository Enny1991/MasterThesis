import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

def read_model_data(filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'res'))
    with open(filename, 'r') as f:
        return pickle.load(f)


def main():
    out, y, inp = read_model_data('19:30_21:10:2015')
    width = 0.35
    plt.figure(figsize=(14, 7))
    for i in range(len(out)):
        chose = i
        plt.bar(np.arange(0-width, 8+width), out[chose], width)
        plt.xticks(np.arange(0-width, 8)+width/2., [str(i) for i in range(0, 9)])        #plt.xlim([0,1])
        plt.grid(which='both')
        plt.title('Suppoose to be at {}'.format(y[chose]))
        plt.show()


if __name__ == '__main__':
    main()

