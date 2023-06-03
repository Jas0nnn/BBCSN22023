
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import openai

cap = cv2.VideoCapture(0)

def getEmotion(data):
    emotiondict = predictions[0]['emotion']
    percentages = [emotiondict[i] for i in (emotiondict)]
    maxpercentage = max(percentages)
    percentageindex = percentages.index(maxpercentage)
    emotion = list(emotiondict)[percentageindex]
    return emotion

def getImage():
    while True:
        ret, frame = cap.read() # Capture frame-by-frame

        if not ret: # if frame is read correctly ret is True
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'): # exits if Q key pressed
            image = cap.read()[1]
            return image

    cap.release()


'''image = getImage()
cv2.imshow("image",image)
cv2.waitKey(0)
predictions = DeepFace.analyze(image)
emotion = getEmotion(predictions)
print(emotion)'''
emotion = "happy"
openai.api_key = "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"


