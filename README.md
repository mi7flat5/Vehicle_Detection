## SDCND **Vehicle Detection Project**

---



[//]: # (Image References)
[image1]: ./car_notcar.jpg
[image2]: ./Hog_Features.jpg
[image3]: ./test1Box.jpg 
[image4]: ./test2Box.jpg 
[image5]: ./test3Box.jpg 
[image6]: ./heatmap.jpg
[image7]: ./boxFromHeatmap.jpg
[image8]: ./boxDrawn.jpg
[image9]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
 
## All code can be found in Image_precessing_classification.ipynb
###Histogram of Oriented Gradients (HOG)

####1.) 
The process began by loading in the vehicle and non_vehicle training data, here is an example of each class:

![alt text][image1] 
I then experemented with different color spaces and parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`: 

 
![alt text][image2] 
These parameters rendered my best results. 
  
####2.) 3.)

I tried a lot of combination but found that most color spaces either cased the classifier to under or over fit by too much, so I settled on YCrCb. I also found that the best results were achived with the smallest pixel per cell that would train in an acceptable amount of time. so I setteld on 8x8 there. I got similar results for any orentation above 9 so I chose the smallest for that result. and Lastsly I chose 2 cells per block to keep the amount of features as small asspossible while achiving acceptable results. The smaller the feature set the quicker training and inferance will be so I wanted to minimize the amount of features for this application.
The code for training the classifier is as follows 
```python
# Feature extraction parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [380, 700] # Min and max in y to search in slide_window()


t = time.time()
car_features = extract_features(car_images, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(noncar_images, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()


X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```
 
###Sliding Window Search

####1.)
I got the best results with using the following scales for the sliding window search. 
```python
scales = [(1.0,400,550),(1.5,400,600),(2.0,400,650),(2.5,400,680)]
for scaler in scales:
    _, boxes = find_cars(test_img, scaler[1], scaler[2], scaler[0], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    box_list += boxes
```
Here is an example of the output: 
 ![alt text][image3]
The code looks like this:
```python
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    boxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return draw_img, boxes
```

####2. Some examples of the classifier along with rejection process for false positives. 
Increment pixels in blank image for each pixel overlapped by a bounding box found by the classifier. 
![alt text][image6]
Filter the heatmap for high value pixels
![alt text][image7]
Apply label and draw rectangles around remaining high confidence pixels. 
![alt text][image8]

This process was augmented in video procesing with a deque, double ended que or stack, of the 8 most recent heatmaps. Each heatmap is added with the current one and then thresholded again for high confidence areas where multiple boxes are found. This results in a finding that must be present in multiple frames and have multiple rectangles found in sliding window search by the classifier. 
---    

### Video Implementation

####1. Link to Project video
Here's a [link to my video result](./project_video_out.mp4)

Here is the code for video processing which includes the deque for filtering false positives and smooting results over 8 frames. 
```python

from collections import deque
heat_list = deque(maxlen = 6)
def process(img):
    
    box_list = []
    for scaler in scales:
        _, boxes = find_cars(img, scaler[1], scaler[2], scaler[0], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list += boxes
    
    heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
   
    heat = apply_threshold(heat, 1)
 
    # Visualize the heatmap when displaying
    current_heatmap = np.clip(heat, 0, 255)
    heat_list.append(current_heatmap)
     
    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in heat_list:
        heatmap = heatmap + heat
        
    heatmap = apply_threshold(heat, 2)
    heatmap = np.clip(heatmap, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    if len(heat_list)< 5:
        return img
    draw_img = draw_labeled_bboxes(img, labels)
    return draw_img
``` 

###Discussion

I found getting the right parameters for this project quite difficult adn I think the best result would come from an agregate of classifications from different color spaces and parameters. This however whould probably be too slow to use in realtime. My implemtation is quite slow, so I think a different appprach all together would be needed. I think that using Convoluational Neural network could be more robust and much quicker  on inference. Or Using a Fully Convolutional Neural Network with Semantic Segmentation might also be a more robust solution. My implementation would likely fail under different lighting conditions.  

