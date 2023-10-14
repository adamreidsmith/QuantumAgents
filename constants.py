import numpy as np

SEED = 1234567899

KET_ZERO = np.array([[1.0], [0.0]])
KET_ZERO.setflags(write=False)
KET_ONE = np.array([[0.0], [1.0]])
KET_ONE.setflags(write=False)

LOG2_INV = 1.0 / np.log(2.0)

I2 = np.eye(2)
I2.setflags(write=False)
M0 = np.array([[1.0, 0.0], [0.0, 0.0]])
M0.setflags(write=False)
M1 = np.array([[0.0, 0.0], [0.0, 1.0]])
M1.setflags(write=False)

QUANTUM = 'quantum'
CLASSICAL = 'classical'
