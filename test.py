import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a=[[[4,5,6,3,6]],[[7,8,9,9,9]]]
    b=[[[5],[6],[3]],[[8],[9],[8]]]
    c=tf.multiply(a,b)
    mm = np.array([1,5,6,3])
    mm[mm > 2] = 0
    pp = 1 - mm
    print(pp)