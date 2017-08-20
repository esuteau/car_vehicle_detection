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
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Computes the HOG features on the input image"""
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

def bin_spatial(img, size=(32, 32)):
    """Resize the input image to the requested output size"""
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Computes color histograms for each channel and concatenate them
       Change default bins_range if reading .png files with mpimg!"""

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True): 
    """Transform the input RGB image to a feature vector depending
    on the chosen input parameters"""

    # Define an empty list to receive features
    img_features = []

    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      

    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append features to list
        img_features.append(spatial_features)

    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append features to list
        img_features.append(hist_features)

    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """Feature extraction, but on a list of image filenames.
    Returns a vector of features for each image"""

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:

        # Read in each one by one
        image = mpimg.imread(file)

        # Get Features
        file_features = single_img_features(
            image, color_space, spatial_size, hist_bins, orient, pix_per_cell,
            cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

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
    notcars = glob.glob('data/non_vehicles/*.png')
    return (cars, notcars)

def run_feature_extraction(cars, notcars, p):
    t=time.time()
    car_features = extract_features(cars,
        color_space=p.color_space,
        orient=p.orient, 
        pix_per_cell=p.pix_per_cell,
        cell_per_block=p.cell_per_block, 
        hog_channel=p.hog_channel,
        spatial_size=p.spatial_size,
        hist_bins=p.hist_bins,
        spatial_feat=p.spatial_feat,
        hist_feat=p.hist_feat,
        hog_feat=p.hog_feat)

    notcar_features = extract_features(notcars, 
        color_space=p.color_space,
        orient=p.orient, 
        pix_per_cell=p.pix_per_cell,
        cell_per_block=p.cell_per_block, 
        hog_channel=p.hog_channel,
        spatial_size=p.spatial_size,
        hist_bins=p.hist_bins,
        spatial_feat=p.spatial_feat,
        hist_feat=p.hist_feat,
        hog_feat=p.hog_feat)

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

    return (scaled_X, y, X_scaler)

def run_classifier(p):
    """Load labeled images, build and train a classifier to different vehicles from non vehicle images."""

    # Import the input images
    print('\n----- Loading Input Dataset -----')
    (cars, notcars) = get_labels_images()
    print('Input Dataset: {} vehicle images and {} non-vehicle images loaded'.format(len(cars), len(notcars)))

    # Reduce the sample size because HOG features are slow to compute
    max_sample_size = min(len(cars), len(notcars))
    sample_size = min(p.sample_size, max_sample_size)
    cars, notcars = cars[0:sample_size], notcars[0:sample_size]
    print('Subset selected: {} vehicles images and {} non-vehicles images'.format(len(cars), len(notcars)))

    # Run Feature Extraction
    print('\n----- Feature Extraction -----')
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

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, p, scale):
    """Compute HOG feature on the whole image, extract the different patches and compute predictions.
    Return a list of rectangle that have been predicted as matching a vehicle"""
    
    draw_img = np.copy(img)

    if p.is_input_jpg:
        img = img.astype(np.float32)/255
    
    # Extract only section to process
    img_tosearch = img[p.ystart:p.ystop,p.xstart:p.xstop,:]

    # Convert and Rescale
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Extract Color Channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // p.pix_per_cell) - p.cell_per_block + 1
    nyblocks = (ch1.shape[0] // p.pix_per_cell) - p.cell_per_block + 1 
    nfeat_per_block = p.orient*p.cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // p.pix_per_cell) - p.cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, p.orient, p.pix_per_cell, p.cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, p.orient, p.pix_per_cell, p.cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, p.orient, p.pix_per_cell, p.cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * p.pix_per_cell
            ytop = ypos * p.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=p.spatial_size)
            hist_features = color_hist(subimg, nbins=p.hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            # If the input patch has been predicted as a car, draw it
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img


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
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)



def process_image(image, p):

    start = time.time()

    draw_image = np.copy(image)
    window_img = np.copy(image)

    # Training was extracted data from .png images (scaled 0 to 1 by mpimg)
    # If the image being searched is jpg (scaled 0 to 255 with mpimg), then we need to rescale it to (0, 1)
    if p.is_input_jpg:
        image = image.astype(np.float32)/255

    # Find all windows to process
    windows = []
    for idx, xy_window in enumerate(p.xy_windows):
        y_start_stop = p.y_start_stops[idx]
        windows.extend(slide_window(image, x_start_stop=p.x_start_stop, y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=p.xy_overlap))

    # Run windows through classifier
    hot_windows = search_windows(image, windows, p.svc, p.X_scaler, color_space=p.color_space, 
                            spatial_size=p.spatial_size, hist_bins=p.hist_bins, 
                            orient=p.orient, pix_per_cell=p.pix_per_cell, 
                            cell_per_block=p.cell_per_block, 
                            hog_channel=p.hog_channel, spatial_feat=p.spatial_feat, 
                            hist_feat=p.hist_feat, hog_feat=p.hog_feat)                       

    window_img = draw_boxes(window_img, hot_windows, color=p.color, thick=p.thick)

    # Create Heat Map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, p.heat_threash)
    print("Max Heat: {}".format(np.max(heat[:])))

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_image, labels, color=p.color, thickness=p.thick)

    elapsed_time = time.time() - start
    print("Processing time: {:.1f} s".format(elapsed_time))

    return (draw_img, heatmap, window_img)

def save_images(draw_img, heatmap, window_img, output_folder, filename):
    # Save Images to file
    window_img_bgr = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
    draw_img_bgr = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)

    out_folder = output_folder + '/window/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, window_img_bgr)

    out_folder = output_folder + '/draw/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, draw_img_bgr)

    out_folder = output_folder + '/heatmap/'
    create_folder(out_folder)
    cv2.imwrite(out_folder + filename, heatmap)


def run_test_image_pipeline():

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
        p.color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        p.orient = 9
        p.pix_per_cell = 8
        p.cell_per_block = 2
        p.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
        p.sample_size = 1000000
        p.spatial_size = (32, 32) # Spatial binning dimensions
        p.hist_bins = 32    # Number of histogram bins
        p.spatial_feat = True # Spatial features on or off
        p.hist_feat = True # Histogram features on or off
        p.hog_feat = True # HOG features on or off
        p.seed = np.random.randint(0, 100)
        p.seed = 1

        # Parameters for Vehicle detection
        p.x_start_stop=[None, None]
        p.xy_windows =      [(256, 256), (128, 128), (64, 64)]
        p.y_start_stops =   [(400, 656), (400, 594), (400, 528)] # Min and max in y to search in slide_window()
        p.xy_overlap=(0.9, 0.9)
        p.color = (0, 0, 255)
        p.thick = 3
        p.heat_threash = 15
        p.is_input_jpg = True

        # Train a classifier on labels images.
        (p.svc, p.X_scaler) = run_classifier(p)

    # Save to file (So that we don't have to train every single time)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load test images
    images_format = 'test_images/*.jpg'
    images = glob.glob(images_format)
    print()

    for idx, image_file in enumerate(images):

        # Read Image
        image = mpimg.imread(image_file)
        
        # Process Image
        (draw_img, heatmap, window_img) = process_image(image, p)

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


def run_video_pipeline():
    from moviepy.editor import VideoFileClip, ImageSequenceClip

    # Define input/output files
    input_video = 'project_video.mp4'
    create_folder('output_images/video/')
    output_dir = 'output_images/video/' + time.strftime("%Y%m%d_%H%M%S") + '/'
    create_folder(output_dir)
    output_video = output_dir + 'output_video.mp4'

    pickle_file = 'parameters.pickle'
    force_overwrite = False
    debug_plot = False

    if os.path.exists(pickle_file) and not force_overwrite:
        print("Reloading parameters and trained classifier from {}".format(pickle_file))
        with open(pickle_file, 'rb') as handle:
            p = pickle.load(handle)

    else:
        # Input parameters
        p = Parameters()
        p.color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        p.orient = 9
        p.pix_per_cell = 8
        p.cell_per_block = 2
        p.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
        p.sample_size = 1000000
        p.spatial_size = (32, 32) # Spatial binning dimensions
        p.hist_bins = 32    # Number of histogram bins
        p.spatial_feat = True # Spatial features on or off
        p.hist_feat = True # Histogram features on or off
        p.hog_feat = True # HOG features on or off
        p.seed = np.random.randint(0, 100)
        p.seed = 1

        # Parameters for Vehicle detection
        p.x_start_stop=[None, None]
        p.xy_windows =      [(256, 256), (128, 128), (64, 64)]
        p.y_start_stops =   [(400, 656), (400, 594), (400, 528)] # Min and max in y to search in slide_window()
        p.xy_overlap=(0.9, 0.9)
        p.color = (0, 0, 255)
        p.thick = 3
        p.heat_threash = 15
        p.is_input_jpg = True

        # Train a classifier on labels images.
        (p.svc, p.X_scaler) = run_classifier(p)

    # Save to file (So that we don't have to train every single time)
    with open(pickle_file, 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load Video
    in_clip = VideoFileClip(input_video)
    nb_frames = int(in_clip.duration * in_clip.fps)
    out_images = []
    for idx, frame in enumerate(in_clip.iter_frames()):

        filename = 'frame_{}.jpg'.format(idx)
        # Save Input Frame for debugging
        output_file = 'output_images/video/project_video_images/' + filename
        cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Process Image
        prog_percent = 100 * (idx+1) / nb_frames
        print("Processing Frame {}/{} {:.1f}%".format(idx+1, nb_frames, prog_percent))
        (draw_img, heatmap, window_img) = process_image(frame, p)

        # Save Images to file
        save_images(draw_img, heatmap, window_img, output_dir, filename)

        # Save Output frame
        output_file = output_dir + filename
        cv2.imwrite(output_file, cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))

        # Store output file
        out_images.append(output_file)

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

    #run_test_image_pipeline()
    run_video_pipeline()