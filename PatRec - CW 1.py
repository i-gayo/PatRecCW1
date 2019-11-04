import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


mat_content = sio.loadmat('face.mat')

print(mat_content)

# Array X contains the face data, each column is one image from a total of 520 images.
# Array l contains the label (face identity) for each image.

