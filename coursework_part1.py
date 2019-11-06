
# Modules
import scipy.io as sio   # Used to load data stored in .mat file
import numpy as np
import matplotlib.pyplot as plt

""" 0. Loading in data: 

X: Contains the facedata, each column represents one face image. 
   Each element in a column is a pixel value for the coordinate of one image. 
l: Contains label (face identity for each image)  

"""

# 0. Load in face dataset
mat_content = sio.loadmat('face.mat')

face_data = mat_content['X']
face_labels = mat_content['l']


""" 1. Split up data to training and testing:
  
    Literature suggests: a split of 60-80 : 40-20 for testing, training respectively 
    to avoid underfitting/over fitting
    Source: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

    Note: Images are ordered in according to their label numbers 
    Need to shuffle the data and the labels to randomize the set and then choose training and datasets 
    
  """

# 1. Initialise training percentage and training numbers
train_pc = 0.8  # Percentage of total data used for training
test_pc = 0.2
no_train = int(train_pc*face_data.shape[1])  # Number of training data: 416
no_test = int(test_pc*face_data.shape[1]) # Number of testing data: 104
# print(no_train, no_test)


def shuffle_data(data, labels):
    """
    :param data: Data in the form of DxN array containing the images in col vectors
    :param labels: Label for each image
    :return: Will return the shuffled dataset and respective labels
    """
    idx = np.random.permutation(data.shape[1])  # Shuffle order of the columns, idx is the shuffled column order
    # print(idx)
    sdata, slabels = data[:, idx], labels.T[idx]  # Rearranges data and labels according to idx, rows in data are same

    return sdata, slabels.T


# 2. Shuffle training data
shuffled_faces, slabels = shuffle_data(face_data, face_labels)

# 3. Partition images for training (416), testing (104)
train_data = shuffled_faces[:, 0:no_train]
test_data = shuffled_faces[:, no_train:]
print(train_data.shape, test_data.shape)

print('Chicken')
""" TESTING SHUFFLING OF DATA:
print('ORIGINAL DATA:')
print(face_data[:,0:5])
shuffled_data = np.random.permutation(face_data.T).T;
#print(np.random.permutation(face_data.T).T)
print('SHUFFLED DATA:')
print(shuffled_data[:,0:5])

#print('CHECKING ORIGNIAL DATA:')
#print(face_data[:,0:5])
"""


""" 2. Apply PCA 
1. Compute average face 
2. Subtract mean face from each face
3. Compute covaraince matrix --> S  = 1/N A*A.T
    Where A contains columns of Xn - X bar for every image
4. Obtain eigen vectors of S 
5. Compute the best M Eigenvectors 
6. Project onto eigenspace ---> Obtain coefficients for each vector
"""


def PCA(training_data):

    # Define cardinality: no of images used: N
    N = training_data.shape[1] # No of cols = no of images N

    # 2.1 Obtain average face
    avg_face = np.mean(training_data, axis=1)  # returns "col vector" of average image
    print(avg_face.shape)
    # plt.imshow(np.reshape(avg_face_numpy, (46, 56)).T, cmap='gist_gray')
    # plt.show()

    # Average face has dimensions (2576, 1) Need to subtract this average face from every image:
    # Subtract average face col vector from each column (representing an image)

    # 2.2 Subtract mean from every face: A = [x1 - x_mean, x2 - x_mean]
    A = training_data - np.reshape(avg_face, (len(avg_face), 1)) # 2nd term extends col. vec. to appropriate matrix
    print('Shape of A is : DxN ', A.shape)

    # 2.3 calculate the covariacne matrix
    S = np.matmul(A, A.T)/N   # Matrix multiplication
    # print(S.shape) S should be DxD dimensional

    # 2.4 Calculate eigenvectors of S
    eigvals, eigvecs = np.linalg.eig(S)  # eigh calculates S for either all real or all comp mat.
    eigvals = (np.round(eigvals, 2)).real
    print(eigvals)   # To get same format as MATLAB
    # NOTE: Some negative and complex part of eigenvalues, however these are negigible and very close to 0!
    # print(eigvals)
    # print(eigvecs)
    plt.plot(np.arange(eigvals.size), np.round(eigvals, 2))
    plt.show()

    # 2.5 Choose M best values of Eigenvectors: choose w best ones

    # 2.6 Project onto eigenspace : obtain w = [an, an2, ..., anm] where an = (x-x_mean).T.ui

    print((np.unique(eigvals)).size)


PCA(train_data)


"""TEST BROADCASTING: Check that this calculation subtracts column wise

x = np.array([[1,2,3], [4,5,6], [7,8,9]])
v = np.array([1, 0, 1])
print((x - np.reshape(v, (3,1))))

"""


def lowdim_PCA(training_data):
    pass





