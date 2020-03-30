import cv2
import imutils
import numpy as np
import tensorflow as tf
import tflearn
from PIL import Image
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

seg = None

# resizing the images 
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(imageName)

#applied run_avg function to get foreground and background of the image
def run_avg(image, aWeight):
    global seg
    if seg is None:
        seg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, seg, aWeight)

#applying segmentation by thresholding and then applying contours to get the presense of object
def segmentation(image, threshold=25):
    diff = cv2.absdiff(seg.astype("uint8"), image)
    ret, threshold_img = cv2.threshold(diff, threshold, 100, cv2.THRESH_BINARY)
    cv2.imshow("threshold image", threshold_img)
    (_, cnts, _) = cv2.findContours(threshold_img.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cnts))
    if len(cnts) <= 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (threshold_img, segmented)

#main function for displaying the thresholded image and prediction on the frames captured from the webcam
def display():
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        img1 = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imshow("Grayscale_image", gray)
        if num_frames < 10:
            run_avg(gray, aWeight)
        else:
            hand = segmentation(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                # cv2.drawContours(
                #     img1, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imwrite('extra.png', thresholded)
                resizeImage('extra.png')
                predictedClass, confidence = getPredictedClass()
                if predictedClass == 0:
                    className = "Swing"
                elif predictedClass == 1:
                    className = "Palm"
                elif predictedClass == 2:
                    className = "Fist"
                if(confidence >= 0.85):
                    cv2.putText(img1, "Pedicted Class : " + className, (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img1, "Confidence : " + str(confidence * 100) + '%',
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(img1, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        cv2.imshow("camera images", img1)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

#extracting class and confidence of the image
def getPredictedClass():
    image = cv2.imread('extra.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))


#architecture of the model for creating an instance for prediction
tf.reset_default_graph()
convnet = input_data(shape=[None, 89, 100, 1], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 256, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 1000, activation='relu')
convnet = dropout(convnet, 0.75)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='regression')
model = tflearn.DNN(convnet, tensorboard_verbose=0)
model.load("TrainedModel/GestureRecogModel.tfl")
display()
