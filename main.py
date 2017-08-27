# Main Python file with all the code necessary to run vehicle detection on a video.

# Here are the different processing blocks to implement (in order)
# - Feature Extraction (Color and Gradient based)
# - Choose and Train Classifier (Linear SVM or other)
# - Import video test images, define bounding box
# - Implement Sliding Window Technique and search for image vehicles
# - Run the pipeline on the video stream

import glob
import os
import pickle
import time
import collections

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import cv2


def get_hog_features(img, orient, pix_per_cell, cell_per_block, transform_sqrt, vis=False, feature_vec=True):
    """Computes the HOG features on the input image"""
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img,
         orientations=orient,
         pixels_per_cell=(pix_per_cell, pix_per_cell),
         cells_per_block=(cell_per_block, cell_per_block),
         transform_sqrt=transform_sqrt, 
         visualise=vis,
         feature_vector=feature_vec,
         block_norm='L2-Hys')

        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=transform_sqrt, 
            visualise=vis,
            feature_vector=feature_vec,
            block_norm='L2-Hys')

        # if len(np.where(np.isnan(features.ravel())) == True):
        #     print("HOG Features problem. Found nan value")
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.show()
                    
        return features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw a bounding box on the input image"""
    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def get_spatial_features(img, size=(32, 32)):
    """Resize the input image to the requested output size"""
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def get_hist_features(img, nbins=32, bins_range=(0, 256)):
    """Computes color histograms for each channel and concatenate them
       Change default bins_range if reading .png files with mpimg!"""

    # Compute the histogram of the color channels separately
    channel1_hist, c1_bin_edges = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist, c2_bin_edges = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist, c3_bin_edges = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))

    # Debug Plot
    if False:
        # Calculating Bin Centers
        bin_centers = (c1_bin_edges[1:]  + c1_bin_edges[0:len(c1_bin_edges)-1])/2

        # Plot
        fig = plt.figure()
        plt.subplot(221)
        plt.barh(bin_centers, channel1_hist)
        plt.title('Channel 1')
        plt.subplot(222)
        plt.barh(bin_centers, channel2_hist)
        plt.title('Channel 2')
        plt.subplot(223)
        plt.barh(bin_centers, channel3_hist)
        plt.title('Channel 3')
        fig.tight_layout()
        plt.show()


    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_features_single_img(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, hist_range=(0,256),
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        transform_sqrt=True): 
    """Transform the input RGB image to a feature vector depending
    on the chosen input parameters"""

    # Define an empty list to receive features
    img_features = []

    # Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, origin='RGB', destination=color_space)

    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = get_spatial_features(img=feature_image, size=spatial_size)
        # Append features to list
        img_features.append(spatial_features)

    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = get_hist_features(img=feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append features to list
        img_features.append(hist_features)

    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_feat = get_hog_features(
                    img=feature_image[:,:,channel], 
                    orient=orient,
                    pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    transform_sqrt=transform_sqrt,
                    vis=False,
                    feature_vec=True)

                hog_features.extend(hog_feat)      
        else:
            hog_features = get_hog_features(
                img=feature_image[:,:,hog_channel],
                orient=orient, 
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                transform_sqrt=transform_sqrt,
                vis=False,
                feature_vec=True)


        # Append features to list
        img_features.append(hog_features)

    # Features
    return np.concatenate(img_features)

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, hist_range=(0,256),
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        Transform_sqrt=True):
    """Feature extraction, but on a list of image filenames.
    Returns a vector of features for each image"""

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for idx, file in enumerate(imgs):

        if idx % 10 == 0:
            print("Feature Extraction: {:.2f}%".format(100.0 * idx / len(imgs)))

        # Read in each one by one
        image = mpimg.imread(file)

        # Get Features
        file_features = get_features_single_img(
            img=image,
            color_space=color_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            orient=orient,
            hist_range=hist_range,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat,
            transform_sqrt=transform_sqrt)

        features.append(file_features)

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
    notcars = glob.glob('data/non_vehicles/Extras/*.png')
    return (cars, notcars)


def get_param_hash(p):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            p.color_space,
            p.orient,
            p.pix_per_cell ,
            p.cell_per_block,
            p.hog_channel,
            p.sample_size,
            p.spatial_size[0],
            p.hist_bins,
            p.hist_range[0],
            p.hist_range[1],
            p.spatial_feat,
            p.hist_feat,
            p.hog_feat,
            p.seed)

def run_feature_extraction(cars, notcars, p):

    t=time.time()

    print("Extracting Features for {} car images".format(len(cars)))
    pickle_name = "temp_save/car_features_{}_{}".format(len(cars), get_param_hash(p))
    if os.path.exists(pickle_name):
        print("Reloading from {}".format(pickle_name))
        with open(pickle_name, 'rb') as f:
            car_features = pickle.load(f)
    else:
        car_features = extract_features(cars,
            color_space=p.color_space,
            orient=p.orient, 
            pix_per_cell=p.pix_per_cell,
            cell_per_block=p.cell_per_block, 
            hog_channel=p.hog_channel,
            spatial_size=p.spatial_size,
            hist_bins=p.hist_bins,
            hist_range=p.hist_range,
            spatial_feat=p.spatial_feat,
            hist_feat=p.hist_feat,
            hog_feat=p.hog_feat,
            Transform_sqrt=p.transform_sqrt)

        with open(pickle_name, 'wb') as f:
            pickle.dump(car_features, f, protocol=pickle.HIGHEST_PROTOCOL)


    print("Extracting Features for {} non-car images".format(len(notcars)))
    pickle_name = "temp_save/notcar_features_{}_{}".format(len(cars), get_param_hash(p))
    if os.path.exists(pickle_name):
        print("Reloading from {}".format(pickle_name))
        with open(pickle_name, 'rb') as f:
            notcar_features = pickle.load(f)
    else:
        notcar_features = extract_features(notcars, 
            color_space=p.color_space,
            orient=p.orient, 
            pix_per_cell=p.pix_per_cell,
            cell_per_block=p.cell_per_block, 
            hog_channel=p.hog_channel,
            spatial_size=p.spatial_size,
            hist_bins=p.hist_bins,
            hist_range=p.hist_range,
            spatial_feat=p.spatial_feat,
            hist_feat=p.hist_feat,
            hog_feat=p.hog_feat,
            tranform_sqrt=transform_sqrt)

        with open(pickle_name, 'wb') as f:
            pickle.dump(notcar_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to run feature extraction on {} car images and {} non-car images.'.format(len(cars), len(notcars)))
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

    return (scaled_X, y, X_scaler)

def run_classifier(p):
    """Load labeled images, build and train a classifier to different vehicles from non vehicle images."""

    # Import the input images
    print('\n----- Loading Input Dataset -----')
    (cars, notcars) = get_labels_images()
    print('Input Dataset: {} vehicle images and {} non-vehicle images loaded'.format(len(cars), len(notcars)))

    # Shuffle datasets
    np.random.seed(p.seed)
    np.random.shuffle(cars)
    np.random.shuffle(notcars)

    # Reduce the sample size because HOG features are slow to compute
    max_sample_size = min(len(cars), len(notcars))
    sample_size = min(p.sample_size, max_sample_size)
    cars, notcars = cars[0:sample_size], notcars[0:sample_size]
    print('Subset selected: {} vehicles images and {} non-vehicles images'.format(len(cars), len(notcars)))

    # Run Feature Extraction
    print('\n----- Feature Extraction -----')
    pickle_file = 'temp_save/classifier_feature_extraction'
    (car_features, notcar_features) = run_feature_extraction(cars, notcars, p)

    # Transform features to X and Y vectors
    (X, y, X_scaler) = prepare_features(car_features, notcar_features)

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=p.seed)
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

    # # Check the prediction time on the test set
    # t=time.time()
    # n_predict = len(X_test)
    # print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    # print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    # t2 = time.time()
    # print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return (svc, X_scaler)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []

    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))    

        # Extract features for that window using get_features_single_img()
        features = get_features_single_img(
            image=test_img,
            color_space=color_space, 
            spatial_size=spatial_size,
            hist_bins=hist_bins, 
            hist_range=hist_range,
            orient=orient,
            pix_per_cell=pix_per_cell, 
            cell_per_block=cell_per_block, 
            hog_channel=hog_channel,
            spatial_feat=spatial_feat, 
            hist_feat=hist_feat,
            hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows

def convert_color(img, origin='RGB', destination='RGB'):
    if origin == destination:
        return np.copy(img)

    if origin == 'RGB':
        if destination == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif destination == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif destination == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif destination == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif destination == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif origin == 'BGR':
        if destination == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif destination == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif destination == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif destination == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif destination == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        raise Exception("Conversion not supported")

    return feature_image

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, svc, scaler, x_start_stop=[None, None], y_start_stop=[None, None], 
                xy_window=(64, 64), color_space='RGB', spatial_size=(32, 32),
                hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True, img_name="", transform_sqrt=True):

    """Compute HOG feature on the whole image, extract the different patches and compute predictions.
    Return a list of rectangle that have been predicted as matching a vehicle"""

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # To save time with the test, we save the save the window detection at the end of this function, so that it can be reloaded later on.
    # This will allow to play with the parameters and iterate at a much faster pace to find the right combination
    output_folder = 'temp_save/'
    create_folder(output_folder)
    save_name = output_folder + '{}_windows_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.p'.format(
        img_name, hog_channel, orient, pix_per_cell, cell_per_block, xy_window[0], color_space, spatial_size[0],
        x_start_stop[0], x_start_stop[1], y_start_stop[0], y_start_stop[1])

    if os.path.exists(save_name):
        with open(save_name, 'rb') as handle:
            on_windows = pickle.load(handle)
            print("Windows reloaded from {}".format(save_name))
            return on_windows

    draw_img = np.copy(img)

    scale = xy_window[0] / 64.0
    
    # Extract only section to process
    img_tosearch = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

    # Convert and Rescale
    ctrans_tosearch = convert_color(img_tosearch, origin='RGB', destination=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Extract Color Channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    xwindow, ywindow = 64,64
    nxcells_per_window = xwindow // pix_per_cell
    nycells_per_window = ywindow // pix_per_cell
    nxblocks_per_window = nxcells_per_window - cell_per_block + 1
    nyblocks_per_window = nycells_per_window - cell_per_block + 1

    # Instead of overlap, define how many cells to step
    cells_per_step = 1 
    nxsteps = (nxblocks - nxblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nyblocks_per_window) // cells_per_step

    if hog_channel == 'ALL':
        t = time.time()
        hog1 = get_hog_features(img=ch1, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False)
        hog2 = get_hog_features(img=ch2, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False)
        hog3 = get_hog_features(img=ch3, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, transform_sqrt=transform_sqrt, feature_vec=False) 
        delay = time.time() - t
        print("HOG Features Computation: {:.2f}s".format(delay))
    else:
        hog_features_full = get_hog_features(
            img=ctrans_tosearch[:,:,hog_channel],
            orient=orient, 
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            transform_sqrt=transform_sqrt,
            feature_vec=False)

    on_windows = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+xwindow, xleft:xleft+ywindow], (64,64))

            # Initialize feature vector
            features = []

            # Compute spatial features if flag is set
            if spatial_feat == True:
                spatial_features = get_spatial_features(img=subimg, size=spatial_size)
                # Append features to list
                features.append(spatial_features)

            # Compute histogram features if flag is set
            if hist_feat == True:
                hist_features = get_hist_features(img=subimg, nbins=hist_bins, bins_range=hist_range)
                # Append features to list
                features.append(hist_features)

            # Extract HOG for this patch
            if hog_feat == True:
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_features_full[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window, :].ravel() 

                features.append(hog_features.ravel())

            # Scale features and make a prediction
            features = np.concatenate(features)
            test_features = scaler.transform(features.reshape(1,-1))    
            test_prediction = svc.predict(test_features)
            
            # If the input patch has been predicted as a car, draw it
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale) + x_start_stop[0]
                ytop_draw = np.int(ytop*scale) + y_start_stop[0]
                xwin_draw = np.int(xwindow*scale)
                ywin_draw = np.int(ywindow*scale)
                on_windows.append([(xbox_left, ytop_draw),(xbox_left + xwin_draw, ytop_draw + ywin_draw)])


    # Save windows on disk, so that it could be reloaded later on.
    with open(save_name, 'wb') as handle:
        pickle.dump(on_windows, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, color, thickness):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thickness)
    # Return the image
    return img

def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def process_image(image, p, find_windows_func):

    start = time.time()

    draw_image = np.copy(image)
    window_img = np.copy(image)

    # Training was extracted data from .png images (scaled 0 to 1 by mpimg)
    # If the image being searched is jpg (scaled 0 to 255 with mpimg), then we need to rescale it to (0, 1)
    if p.is_input_jpg:
        image = image.astype(np.float32)/255

    # Find car in windows
    hot_windows = find_windows_func(image, p)

    # Draw Boxes
    window_img = draw_boxes(window_img, hot_windows, color=p.color, thick=p.thick)

    # Create Heat Map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, p.heat_thresh)
    print("Max Heat: {}".format(np.max(heat[:])))

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels, color=p.color, thickness=p.thick)

    elapsed_time = time.time() - start
    print("Processing time: {:.1f} s".format(elapsed_time))

    return (draw_img, heatmap, window_img)



def process_image_with_memory(image, p, find_windows_func, raw_heat_maps, img_name):
    # Same image processing as above, but with memory, i.e. we combine
    # heatmaps from previous frames, calculate the average heatmap
    # and threshold the average result.
    # Doing that helps removing false positives

    start = time.time()

    draw_image = np.copy(image)
    window_img = np.copy(image)

    # Training was extracted data from .png images (scaled 0 to 1 by mpimg)
    # If the image being searched is jpg (scaled 0 to 255 with mpimg), then we need to rescale it to (0, 1)
    if p.is_input_jpg:
        image = image.astype(np.float32)/255

    # Find car in windows
    hot_windows = find_windows_func(image, p, img_name=img_name)

    # Draw Boxes
    window_img = draw_boxes(window_img, hot_windows, color=p.color, thick=p.thick)

    # Create Heat Map
    heat_raw = np.zeros_like(image[:,:,0]).astype(np.float)
    heat_raw = add_heat(heat_raw,hot_windows)

    # Now that we have detected the current heat map, we are going to use memory to detect which blobs should be worth considering.
    # Instead of combining the current frame with the X previous ones and doing an average, we are going to use the X previous frames to
    # define which blobs to keep in the current frame. To do that will penalize areas where not blobs were detected in the previous frames.
   
    #heat_raw[heat_raw == 0] = -5

    # Now calculate an average heat map
    if len(raw_heat_maps) > 0:
        heat_tot = np.dstack(raw_heat_maps)
        heat_tot = np.dstack((heat_tot, heat_raw))
        heat_avg = np.sum(heat_tot, axis=2)
    else:
        heat_avg = heat_raw

    # Apply threshold to help remove false positives
    theshold = (len(raw_heat_maps) + 1) * p.heat_thresh
    print("Max Heat: {} (Threshold: {})".format(np.max(heat_avg[:]), theshold))
    heat = apply_threshold(heat_avg, theshold)
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels, color=p.color, thickness=p.thick)

    elapsed_time = time.time() - start
    print("Processing time: {:.1f} s".format(elapsed_time))

    return (draw_img, heatmap, window_img, heat_raw, heat_avg)



def process_image_fast(image, p):
    return process_image(image, p, find_windows_fast)

def process_image_slow(image, p):
    return process_image(image, p, find_windows_slow)


def find_windows_fast(image, p, img_name=""):
    # Find the windows in the image that represent a car.
    # Calculate the HOG features on the entire image and then extract the windows.

    hot_windows = []
    for idx in range(len(p.y_start_stops)):
        win = find_cars(
            img=image,
            svc=p.svc,
            scaler=p.X_scaler,
            x_start_stop=p.x_start_stops[idx],
            y_start_stop=p.y_start_stops[idx],
            xy_window=p.xy_windows[idx],
            color_space=p.color_space,
            spatial_size=p.spatial_size,
            hist_bins=p.hist_bins,
            hist_range=p.hist_range,
            orient=p.orient,
            pix_per_cell=p.pix_per_cell,
            cell_per_block=p.cell_per_block,
            hog_channel=p.hog_channel,
            spatial_feat=p.spatial_feat,
            hist_feat=p.hist_feat,
            hog_feat=p.hog_feat,
            img_name=img_name,
            transform_sqrt=p.transform_sqrt
            )

        hot_windows.extend(win)

    return hot_windows


def find_windows_slow(image, p):
    # Find the windows in the image that represent a car.
    # Slower than the fast method as it computes HOG features on all the smaller windows
    # Since some of those windows overlap, we end up recalculating the same features.

    # Find all windows to process
    windows = []
    for idx, xy_window in enumerate(p.xy_windows):
        y_start_stop = p.y_start_stops[idx]
        windows.extend(slide_window(image, x_start_stop=p.x_start_stops[idx], y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=p.xy_overlap))

    # Run windows through classifier
    hot_windows = search_windows(image, windows, p.svc, p.X_scaler, color_space=p.color_space, 
                            spatial_size=p.spatial_size, hist_bins=p.hist_bins, 
                            orient=p.orient, pix_per_cell=p.pix_per_cell, 
                            cell_per_block=p.cell_per_block, 
                            hog_channel=p.hog_channel, spatial_feat=p.spatial_feat, 
                            hist_feat=p.hist_feat, hog_feat=p.hog_feat)             

    return hot_windows

def save_images(draw_img, heatmap, heat_avg, window_img, output_folder, filename):
    # Save Images to file
    window_img_bgr = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
    draw_img_bgr = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    heat_avg = np.clip(heat_avg, 0, 255).astype(np.uint8)
    heatmap = heatmap.astype(np.uint8)
    
    out_folder = output_folder + '/window/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, window_img_bgr)

    out_folder = output_folder + '/draw/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, draw_img_bgr)

    out_folder = output_folder + '/heatmap/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, heatmap)

    out_folder = output_folder + '/heatavg/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, heat_avg)


def run_test_image_pipeline(image_processing_func=process_image):

    pickle_file = 'parameters.pickle'
    output_folder = 'output_images/'
    force_overwrite = False
    debug_plot = False

    if os.path.exists(pickle_file) and not force_overwrite:
        print("Reloading parameters and trained classifier from {}".format(pickle_file))
        with open(pickle_file, 'rb') as handle:
            p = pickle.load(handle)

    else:
        # Input parameters
        p = Parameters()
        p.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        p.orient = 9
        p.pix_per_cell = 8
        p.cell_per_block = 2
        p.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        p.sample_size = 1000000
        p.spatial_size = (32, 32) # Spatial binning dimensions
        p.hist_bins = 32    # Number of histogram bins
        p.spatial_feat = True # Spatial features on or off
        p.hist_feat = True # Histogram features on or off
        p.hog_feat = True # HOG features on or off
        p.transform_sqrt = True
        p.seed = np.random.randint(0, 100)
        p.seed = 1

        # Parameters for Vehicle detection
        p.xy_windows =      [[192, 192], [128, 128], [96, 96], [64, 64]]
        p.y_start_stops =   [[400, 656], [400, 592], [400, 560], [400, 528]] # Min and max in y to search in slide_window[]
        p.x_start_stops =   [[None, None], [None, None], [None, None], [None, None]]
        p.xy_overlap=(0.9, 0.9)
        p.color = (0, 0, 255)
        p.thick = 3
        p.heat_thresh = 15
        p.is_input_jpg = True

        # Train a classifier on labels images.
        (p.svc, p.X_scaler) = run_classifier(p)

    p.heat_thresh = 8
    p.xy_windows =      [[192, 192], [128, 128], [96, 96], [64, 64]]
    p.y_start_stops =   [[400, 656], [400, 592], [400, 560], [400, 528]] # Min and max in y to search in slide_window[]

    # Save to file (So that we don't have to train every single time)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load test images
    images_format = 'test_images/*.jpg'
    images = glob.glob(images_format)

    for idx, image_file in enumerate(images):

        # Read Image
        image = mpimg.imread(image_file)
        
        # Process Image
        (draw_img, heatmap, window_img) = image_processing_func(image, p)

        # Save Images to file
        filename = os.path.basename(image_file)
        save_images(draw_img, heatmap, window_img, output_folder, filename)

        # Debug Plot
        if debug_plot:
            fig = plt.figure()
            plt.subplot(131)
            plt.imshow(window_img)
            plt.title('Boxes')
            plt.subplot(132)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(133)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()

            plt.show()


def run_video_pipeline(image_processing_func):
    from moviepy.editor import VideoFileClip, ImageSequenceClip

    # Define input/output files
    input_video = 'project_video.mp4'
    create_folder('output_images/video/')
    output_dir = 'output_images/video/' + time.strftime("%Y%m%d_%H%M%S") + '/'
    create_folder(output_dir)
    output_video = output_dir + 'output_video.mp4'

    pickle_file = 'parameters.pickle'
    force_overwrite = True
    debug_plot = False

    if os.path.exists(pickle_file) and not force_overwrite:
        print("Reloading parameters and trained classifier from {}".format(pickle_file))
        with open(pickle_file, 'rb') as handle:
            p = pickle.load(handle)

    else:
        # Input parameters
        p = Parameters()
        p.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        p.orient = 11
        p.pix_per_cell = 8
        p.cell_per_block = 2
        p.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        p.sample_size = 100
        p.spatial_size = (32, 32) # Spatial binning dimensions
        p.hist_bins = 32    # Number of histogram bins
        p.hist_range = (0.0, 1.0)
        p.spatial_feat = False # Spatial features on or off
        p.hist_feat = True # Histogram features on or off
        p.hog_feat = True # HOG features on or off
        p.transform_sqrt = True
        p.seed = np.random.randint(0, 100)
        p.seed = 1
        p.save_images = False

        # Parameters for Vehicle detection
        p.xy_windows =      [[64, 64]]
        p.y_start_stops =   [[350, 656]] # Min and max in y to search in slide_window[]
        p.x_start_stops =   [[None, None]]
        p.xy_overlap=(0.9, 0.9)
        p.color = (0, 0, 255)
        p.thick = 3
        p.heat_thresh = 15
        p.is_input_jpg = True
        p.frame_memory_length = 40

        # Train a classifier on labels images.
        (p.svc, p.X_scaler) = run_classifier(p)

    # Save to file (So that we don't have to train every single time)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    p.heat_thresh = 20
    p.frame_memory_length = 10
    p.save_images = True
    p.xy_windows =      [[64, 64]]
    p.y_start_stops =   [[400, 656]] # Min and max in y to search in slide_window[]
    p.x_start_stops =   [[None, None]]
    # p.xy_windows =      [[128, 128], [64, 64]]
    # p.y_start_stops =   [[400, None],  [400, None]] # Min and max in y to search in slide_window[]
    # p.x_start_stops =   [[None, None],  [350, None]]


    # Load Video
    in_clip = VideoFileClip(input_video).subclip(13,18)
    nb_frames = int(in_clip.duration * in_clip.fps)
    out_images = []
    heat_map_memory = collections.deque(maxlen=p.frame_memory_length)
    for idx, frame in enumerate(in_clip.iter_frames()):

        filename = 'frame_{}.jpg'.format(idx)
        # Save Input Frame for debugging
        output_file = 'output_images/video/project_video_images/' + filename
        cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Process Image
        prog_percent = 100 * (idx+1) / nb_frames
        print("Processing Frame {}/{} {:.1f}%".format(idx+1, nb_frames, prog_percent))
        (draw_img, heatmap, window_img, raw_heat_map, heat_avg) = process_image_with_memory(image=frame,
                                                                                            p=p,
                                                                                            find_windows_func=find_windows_fast,
                                                                                            raw_heat_maps=heat_map_memory,
                                                                                            img_name="frame_{}".format(idx))
        # Append new map to queue
        heat_map_memory.append(raw_heat_map)

        # Save Images to file
        if p.save_images:
            save_images(draw_img, heatmap, heat_avg, window_img, output_dir, filename)

        # Save Output frame
        output_file = output_dir + filename
        cv2.imwrite(output_file, cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))

        # Store output file
        out_images.append(output_file)

        # Debug Plot
        if debug_plot:
            fig = plt.figure()
            plt.subplot(221)
            plt.imshow(window_img)
            plt.title('Boxes')
            plt.subplot(222)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(223)
            plt.imshow(raw_heat_map, cmap='hot')
            plt.title('Heat Map')
            plt.subplot(224)
            plt.imshow(heat_avg, cmap='hot')
            plt.title('Heat Map Average')
            fig.tight_layout()

            plt.show()

    # Make a clip
    print("Writing output video at {}".format(output_video))
    out_clip = ImageSequenceClip(out_images, fps=24)
    out_clip.write_videofile(output_video, audio=False)

if __name__ == "__main__":

    print('\n----------------------------------------------')
    print('Udacity Self Driving Nano-degree')
    print('Project 5: Vehicle Detection')
    print('Author: Erwan Suteau')
    print('----------------------------------------------')

    t = time.time()
    #run_test_image_pipeline(process_image_fast)
    run_video_pipeline(process_image_fast)
    total = time.time() - t
    print('\nCompleted in {:.2f}s'.format(total))