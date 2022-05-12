import cv2
import numpy as np
import os

captureVid = cv2.VideoCapture('Video2.mp4');
widthHeight = 320
confThreshold = 0.5
nms_threshold = 0.3

objectFile = 'names'
objectTypes = []
with open(objectFile, 'rt') as f:
    objectTypes = f.read().split('\n')

modelConfigs = 'C:\\Users\\Thinira\\PycharmProjects\\ObjectTracking\\yolov3-tiny.cfg'
modelWeights = 'C:\\Users\\Thinira\\PycharmProjects\\ObjectTracking\\yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfigs, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT , wT, cT = img.shape
    bbox =[]
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence >confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y =int(det[0]*wT-w/2), int(det[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold)
    for i in indices:

        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255),2)
        cv2.putText(img,f'{objectTypes[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
while True:
    success, img = captureVid.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop= False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    try:
        outputs = net.forward(outputNames)
    except KeyboardInterrupt:
        continue


    findObjects(outputs,img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
