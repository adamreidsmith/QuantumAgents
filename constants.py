import numpy as np

SEED = 1234567899

KET_ZERO = np.array([[1.0], [0.0]])
KET_ZERO.setflags(write=False)
KET_ONE = np.array([[0.0], [1.0]])
KET_ONE.setflags(write=False)
