import cv2
import numpy as np
import json, operator
from tqdm import tqdm 
import glob 
import pandas as pd

#load the json file
bbox_data = json.load(open('frame_bbox_data.json'))

#create pandas dataframe
df = pd.DataFrame(bbox_data)
print(df)
#sort the data frame
df_sorted = df.sort_values(by='frames')
print(df_sorted)


#data frame legend
# print(df_sorted['frames'])
# print(df_sorted['bboxes'])
# print(df_sorted.iloc[0,0])
# print(df_sorted.iloc[2,1])
# print(df_sorted.iloc[2,1][0])
# print(df_sorted.iloc[2,1][1])
# print(df_sorted.iloc[2,1][2])
# print(df_sorted.iloc[2,1][3])
df_len = len(df_sorted)
 


class Tracker:
    """
    This class represents a tracker object that uses OpenCV and Kalman Filters.
    """

    def __init__(self, id, hsv_frame, track_window):
        """
        Initializes the Tracker object.

        Args:
            id (int): Identifier for the tracker.
            hsv_frame (numpy.ndarray): HSV frame.
            track_window (tuple): Tuple containing the initial position of the tracked object (x, y, width, height).
        """
        self.id = id
        self.track_window = track_window
        self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

        # #load data from json file
        # bbox_data = json.load(open('bbox_data.json'))
        
        # print(len(bbox_data))
        # for clip_id, clip in tqdm(enumerate(bbox_data)):
        #     print(clip["frames"], clip["bboxes"])
        #     # for bbox_id, bbox in enumerate(clip["bboxes"]):
        #     #   print()

        # Initialize the histogram.
        x, y, w, h = track_window
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16], [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Create a Kalman filter object with 4 state variables and 2 measurement variables.
        self.kalman = cv2.KalmanFilter(4, 2)

        # Set the measurement matrix of the Kalman filter.
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

        # Set the transition matrix of the Kalman filter.
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        # Set the process noise covariance matrix of the Kalman filter.
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03

        cx = x + w / 2
        cy = y + h / 2

        # Set the initial predicted state of the Kalman filter.
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)

        # Set the corrected state of the Kalman filter.
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    def update(self, frame, hsv_frame):
        """
        Updates the Tracker object.

        Args:
            frame (numpy.ndarray): Current frame.
            hsv_frame (numpy.ndarray): HSV frame.
        """
        # Predict the new location of the tracked object.
        prediction = self.kalman.predict()

        # Perform meanshift to get the new location.
        ret, self.track_window = cv2.meanShift(self.roi_hist, self.track_window, self.term_crit)

        # Extract the new location of the tracked object.
        x, y, w, h = self.track_window
        new_roi = hsv_frame[y:y+h, x:x+w]

        # Update the histogram.
        new_roi_hist = cv2.calcHist([new_roi], [0, 2], None, [15, 16], [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(new_roi_hist, new_roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Update the Kalman filter with the new measurement.
        measurement = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
        self.kalman.correct(measurement)

        # Draw the predicted bounding box.
        x_pred, y_pred = int(prediction[0]), int(prediction[1])
        cv2.rectangle(frame, (x_pred - w // 2, y_pred - h // 2), (x_pred + w // 2, y_pred + h // 2), (0, 255, 0), 2)

# Open the video file.
cap = cv2.VideoCapture('cricket_video.mp4')

# Create the KNN background subtractor.
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Set the history length for the background subtractor.
history_length = 20
bg_subtractor.setHistory(history_length)

# Create kernel for erode and dilate operations.
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))

# Create an empty list to store the tracked senators.
senators = []

# Counter to keep track of the number of history frames populated.
num_history_frames_populated = 0

# Start processing each frame of the video.
while True:
    # Read the current frame from the video.
    grabbed, frame = cap.read()

    #toget number of frames
    frame_number  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    set_frame_number = frame_number-1
    # print('Frame_number', frame_number)

    #json file check
    for i in range(df_len):
        # print('suck')
        # print(frame_number)
        # print(df_sorted.iloc[i,0])
        if set_frame_number ==df_sorted.iloc[i,0]:
            # print(df_sorted.iloc[i,0])
            print('fuck', i)
            v_x= df_sorted.iloc[i,1][0]
            v_y= df_sorted.iloc[i,1][1]
            v_w= df_sorted.iloc[i,1][2]
            v_h= df_sorted.iloc[i,1][3]

            print(v_x, v_y, v_w, v_h)


    # If there are no more frames to read, break out of the loop.
    if not grabbed:
        break

    # Apply the KNN background subtractor to get the foreground mask.
    fg_mask = bg_subtractor.apply(frame)

    # Let the background subtractor build up a history before further processing.
    if num_history_frames_populated < history_length:
        num_history_frames_populated += 1
        continue

    # Create the thresholded image using the foreground mask.
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    # Perform erosion and dilation to improve the thresholded image.
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    # Find contours in the thresholded image.
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the frame to HSV color space for tracking.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw red rectangles around large contours.
    # If there are no senators being tracked yet, create new trackers.
    should_initialize_senators = len(senators) == 0
    id = 0
    for c in contours:
        # Check if the contour area is larger than a threshold.
        if cv2.contourArea(c) > 500:
            # Get the bounding rectangle coordinates.
            (x, y, w, h) = cv2.boundingRect(c)
            
            #midpoints of the contours
            X=(x+w)/2
            Y=(y+h)/2

            # Draw a rectangle around the contour.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # Convert ID into str
            id_str = str(id)
            cv2.putText(frame, id_str , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # If no senators are being tracked yet, create a new tracker for each contour.
            if should_initialize_senators:
                senators.append(Tracker(id, hsv_frame, (x, y, w, h)))    

            id += 1

    # Update the tracking of each senator.
    for senator in senators:
        senator.update(frame, hsv_frame)

    # Display the frame with senators being tracked.
    cv2.imshow('Senators Tracked', frame)

    # Wait for the user to press a key (110ms delay).
    k = cv2.waitKey(110)

    # If the user presses the Escape key (key code 27), exit the loop.
    if k == 27:
        break
