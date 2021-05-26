import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl

#avoiding security issue
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#loading the images and labels
X = np.load('C 123- Alphabet Recognition.npz')['arr_0']

y = pd.read_csv("C 123- Alphabet Labels Data.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

nclasses = len(classes)

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

#scaling the values so that they are all between 0 and 1
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#fitting the the training data into the multinomial logistic regression classifier
classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

#getting the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the prediction model is -->", accuracy)

#looping for every frame in the video
video = cv2.VideoCapture(0)
while(True):
    try:
        #reading every frame
        ret, frame = video.read()

        #converting the frame to gray color
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #getting the height and width of the frame
        height, width = grayscale.shape

        #top left and bottom right points to draw the rectangle
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(grayscale, upper_left, bottom_right, (0, 255, 0), 2)

        #getting the area of the rectangle
        region_of_interest = grayscale[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        #converting the region of interest into PIL format
        pillow_image = Image.fromarray(region_of_interest)

        #converting the PIL image into gray color by passing L as the parameter
        grayscale_image = pillow_image.convert("L")

        #resizing the image as 22 and 30 because 22 x 30 = 660 = the number of pixels in one image
        resized_image = grayscale_image.resize((22, 30), Image.ANTIALIAS)

        #inverting the image for accurate prediction
        inverted_image = PIL.ImageOps.invert(resized_image)

        #filtering the pixels
        pixel_filter = 20

        #getting minimum pixel value by getting percentile
        min_pixel = np.percentile(inverted_image, pixel_filter)

        #making all the values between 0 and 255
        scaled_image = np.clip(inverted_image-min_pixel, 0, 255)

        #getting the maximum pixel value
        max_pixel = np.max(inverted_image)

        #scaling the image by dividing by the maximum pixel, so that all the values are between 0 and 1
        scaled_image = np.asarray(scaled_image)/max_pixel

        #taking a sample with shape 660 because 20 x 30 = 660
        sample = np.array(scaled_image).reshape(1, 660)
        
        #predicting the class
        prediction = classifier.predict(sample)

        print("The predicted alphabet is -->", prediction)

        #displaying the video
        cv2.imshow('Alphabet', grayscale)
        
        #waiting for a key to be pressed to end the program
        key = cv2.waitKey(1)

        #breaking the loop when q is pressed
        if key & 0xFF == ord('q'):
            break

    #breaking the loop when esc is pressed
        if key & 0xFF == 27:
            break

    #exception handling
    except Exception as e:
        pass

#closing the windows and cam
video.release()
cv2.destroyAllWindows()