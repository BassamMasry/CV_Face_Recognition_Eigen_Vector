import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

dataset_path = os.getcwd()+'/Dataset'# path to the dataset
test_image_path = "angelina-jolie.jpg"
threshold = 3000


# def plot(images, num_rows, num_cols):
#     plt.figure(figsize=(2*num_cols,2*num_rows))
#     for i in range(num_rows*num_cols):
#         plt.subplot(num_rows,num_cols,i+1, frame_on = False, xticks = [], yticks = [])
#         plt.imshow(images[i], cmap=plt.cm.gray)
#     plt.tight_layout()

# ____________________________________________________

num_images = 0
names_list = []
root_list = []

for root, dirs, files in os.walk(dataset_path):
    file_list = []
    for file in files:
        if file[-3:] == 'pgm' or file[-3:] == 'jpg':
            file_list.append(file)
            num_images += 1
    names_list.append(file_list)
    root_list.append(root)

shape = None
# Assuming fixed shape for all images
for i in range(len(names_list)):
    if(len(names_list[i]) != 0):
        img = cv2.imread(root_list[i] + "/" + names_list[i][0], 0)
        shape = img.shape
        break

img_matrix = np.zeros((num_images,shape[0],shape[1]),dtype = np.float64)
counter = 0
for i1 in range(len(names_list)):
    if(len(names_list[i1]) != 0):
        for i2 in range(len(names_list[i1])):
            img = cv2.imread(root_list[i1] + "/" + names_list[i1][i2], 0)
            # Make sure the image is the size we want
            img = cv2.resize(img,(shape[1],shape[0]))
            img_matrix[counter] = img
            counter += 1

# plot(img_matrix, 5, 5)

# ____________________________________________________

mean_vector = np.mean(img_matrix, axis = 0, dtype = np.float64).reshape(1,shape[0], shape[1])
mean_shifted = img_matrix - mean_vector
mean_shidted2D = mean_shifted.view()
mean_shidted2D.shape = (num_images, shape[0] * shape[1])

# plt.imshow(np.resize(mean_vector,(shape[0],shape[1])),plt.cm.gray)
# plt.title('Mean Image')
# plt.show()

# plot(mean_shifted, 5, 5)

# ____________________________________________________

# Creating a symmetric matrix
L = np.dot(mean_shidted2D,  mean_shidted2D.transpose())/num_images
eigenvalues,eigenvectors = np.linalg.eig(L)
# Get reverese-order index
idx = eigenvalues.argsort()[::-1]
# sort eigenvalues in descending order
eigenvalues = eigenvalues[idx]
# sort eigenvectors in descending order
eigenvectors = eigenvectors[:,idx]
# perform matrix multiplication, each column is an eigenvector
eigenvector_C = np.matmul(mean_shidted2D.transpose(), eigenvectors)
# Noemalize the vector
# np.linalg.norm(eigenvector_C.transpose(), axis = 1).reshape(num_images, 1) is the array of norms
eigenfaces = eigenvector_C.transpose() / np.linalg.norm(eigenvector_C.transpose(), axis = 1).reshape(num_images, 1)

# ____________________________________________________

eigenface_labels = np.arange(num_images)
# plot(eigenfaces.reshape(num_images, shape[0], shape[1]), 5, 5)

# ____________________________________________________

test_img = cv2.imread(test_image_path, 0)
test_img = cv2.resize(test_img,(shape[1],shape[0]))
test_img = test_img - mean_vector.reshape(shape[0], shape[1])
test_img = test_img.reshape(shape[0] * shape[1])

E = eigenfaces.dot(test_img)

flatten_names = []
for index, item in enumerate(names_list):
    for element in item:
        flatten_names.append(root_list[index] + "/" + element)

smallest_value =None
index = None 
for idx in range(num_images):
    E_mean = np.dot(eigenfaces, mean_shidted2D[idx])
    diff = E-E_mean
    distance = np.sqrt(np.dot(diff,diff))
    if smallest_value==None:
        smallest_value=distance
        index = idx
    if smallest_value>distance:
        smallest_value=distance
        index=idx
if smallest_value<threshold:
    print(smallest_value,flatten_names[index])
else:
    print(smallest_value,"unknown Face")