import cv2
import numpy as np


class Tracker():
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

        # Initialize the histogram.
        x, y, w, h = track_window
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16],[0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Create a Kalman filter object with 4 state variables and 2 measurement variables.
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Set the measurement matrix of the Kalman filter.
        # It defines how the state variables are mapped to the measurement variables.
        # In this case, the measurement matrix is a 2x4 matrix that maps the x and y position measurements to the state variables.
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

        # Set the transition matrix of the Kalman filter.
        # It defines how the state variables evolve over time.
        # In this case, the transition matrix is a 4x4 matrix that represents a simple linear motion model.
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        # Set the process noise covariance matrix of the Kalman filter.
        # It represents the uncertainty in the process model and affects how the Kalman filter predicts the next state.
        # In this case, the process noise covariance matrix is a diagonal matrix scaled by 0.03.
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03

        cx = x+w/2
        cy = y+h/2
        
        # Set the initial predicted state of the Kalman filter.
        # It is a 4x1 column vector that represents the initial estimate of the tracked object's state.
        # The first two elements are the predicted x and y positions, initialized to the center of the tracked window.
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        
        # Set the corrected state of the Kalman filter.
        # It is a 4x1 column vector that represents the current estimated state of the tracked object.
        # Initially, it is set to the same value as the predicted state.
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)



# Open the video file.
cap = cv2.VideoCapture('sample_video.mp4')

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
            
            # Draw a rectangle around the contour.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
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





