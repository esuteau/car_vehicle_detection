# Main Python file with all the code necessary to run vehicle detection on a video.

# Here are the different processing blocks to implement (in order)
# - Feature Extraction (Color and Gradient based)
# - Choose and Train Classifier (Linear SVM or other)
# - Import video test images, define bounding box
# - Implement Sliding Window Technique and search for image vehicles
# - Run the pipeline on the video stream

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                    visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

class Parameters:
    def __init__(self):
        return

def get_labels_images():
    """Find all images that will be used to train and validate the classifier"""
    cars = glob.glob('data/vehicles/GTI_Far/*.png') + \
           glob.glob('data/vehicles/GTI_Left/*.png') + \
           glob.glob('data/vehicles/GTI_MiddleClose/*.png') + \
           glob.glob('data/vehicles/GTI_Right/*.png') + \
           glob.glob('data/vehicles/KITTI_extracted/*.png')
    notcars = glob.glob('data/non_vehicles/*.png')
    return (cars, notcars)

def run_feature_extraction(cars, notcars, p):
    t=time.time()
    car_features = extract_features(cars, cspace=p.colorspace, orient=p.orient, 
                            pix_per_cell=p.pix_per_cell, cell_per_block=p.cell_per_block, 
                            hog_channel=p.hog_channel)
    notcar_features = extract_features(notcars, cspace=p.colorspace, orient=p.orient, 
                            pix_per_cell=p.pix_per_cell, cell_per_block=p.cell_per_block, 
                            hog_channel=p.hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    return (car_features, notcar_features)

def prepare_features(car_features, notcar_features):
    """Reshape feature to build X and Y arrays, ready for the classifier"""
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return (scaled_X, y)

def run_classifier():
    """Load labeled images, build and train a classifier to different vehicles from non vehicle images."""

    # Import the input images
    print('\n----- Loading Input Dataset -----')
    (cars, notcars) = get_labels_images()
    print('Input Dataset: {} vehicle images and {} non-vehicle images loaded'.format(len(cars), len(notcars)))

    # Input parameters
    p = Parameters()
    p.colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    p.orient = 9
    p.pix_per_cell = 8
    p.cell_per_block = 2
    p.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    p.sample_size = min(len(cars), len(notcars))
    p.sample_size = 500

    # Reduce the sample size because HOG features are slow to compute
    cars, notcars = cars[0:p.sample_size], notcars[0:p.sample_size]
    print('Subset selected: {} vehicles images and {} non-vehicles images'.format(len(cars), len(notcars)))

    # Run Feature Extraction
    print('\n----- Feature Extraction -----')
    (car_features, notcar_features) = run_feature_extraction(cars, notcars, p)

    # Transform features to X and Y vectors
    (X, y) = prepare_features(car_features, notcar_features)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    print('Using:',p.orient,'orientations',p.pix_per_cell, 'pixels per cell and', p.cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Fit a linear SVC
    print('\n----- Classifier -----')
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Validate classifier on Test set. Print accuracy
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time on the test set
    # t=time.time()
    # n_predict = len(X_test)
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    # t2 = time.time()
    # print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc

if __name__ == "__main__":

    print('\n----------------------------------------------')
    print('Udacity Self Driving Nano-degree')
    print('Project 5: Vehicle Detection')
    print('Author: Erwan Suteau')
    print('----------------------------------------------')

    # Train a classifier on labels images.
    svc = run_classifier()