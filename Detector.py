import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import subprocess
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from random import randint

np.random.seed(20)

class Detector:
    def __init__(self, modelURL, classFile):
        self.model = None
        self.classesList = None
        self.colorList = None
        self.modelName = None
        self.cacheDir = None

        self.downloadModel(modelURL)
        self.loadModel()
        self.readClasses(classFile)

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.tar.gz')]

        self.cacheDir = "./Our_model"
        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName, origin=modelURL,
                 cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model " + self.modelName + " loaded successfully...")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = tf.convert_to_tensor(image[tf.newaxis, ...], dtype=tf.uint8)
        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, _ = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex].upper()
                print("classLabelText   ",classLabelText)
                classColor = self.colorList[classIndex]
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                print("displayText   ",displayText, "   ++++ ", classColor)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                ########################################
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin)*0.2))

                cv2.line(image, (xmin,ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmin,ymin), (xmin , ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax,ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmax,ymin), (xmax , ymin + lineWidth), classColor, thickness=5)
        
                ########################################
                
                cv2.line(image, (xmin,ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin,ymax), (xmin , ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax,ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax,ymax), (xmax , ymax - lineWidth), classColor, thickness=5)



        return image

    def predictImage(self, imagePath, threshold=0.5, save_path=None):
        if save_path is None:
            save_path = os.path.join("result", "images")

        os.makedirs(save_path, exist_ok=True)

        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image, threshold)
        output_path = os.path.join(save_path, os.path.basename(imagePath)[:-4] + "_result.jpg")
        cv2.imwrite(output_path, bboxImage)

        # Display the result in a popup window
        self.showImagePopup(bboxImage)

    def predictVideoSource(self, videoPath, threshold=0.5, save_path=None, process_every_nth_frame=12):
        if save_path is None:
            save_path = os.path.join("result", "videos")

        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening video file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(save_path, os.path.basename(videoPath)[:-4] + "_result.avi")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % process_every_nth_frame != 0:
                continue

            bboxFrame = self.createBoundingBox(frame, threshold)
            video_writer.write(bboxFrame)
            cv2.imshow('RTSP Stream Detection', bboxFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        print("Output saved at:", output_path)

    def predictRTSPStream(self,channel, threshold=0.5, process_every_nth_frame=12,save_path=None):
        ch=str(channel)
        rtspStreamURL = "rtsp://admin:DK@admin85@172.31.37.125:554/cam/realmonitor?channel="+ch+"&subtype=0"
        threshold = 0.5

        if save_path is None:
            save_path = os.path.join("result", "videos")

        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(rtspStreamURL)
        # cap = cv2.VideoCapture(videoPath)
        

        if not cap.isOpened():
            print("Error opening video file")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        b=str(randint(1, 1000))
        c=str(randint(1, 1000))
        d=str(randint(1, 1000))
        
        output_path = os.path.join(save_path, rtspStreamURL[61:]+b+c+d+ "_result.avi")
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        # if not cap.isOpened():
        #     print("Error opening video stream or file")
        #     return

        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Increment frame counter
            frame_counter += 1
            if frame_counter % process_every_nth_frame != 0:
                continue

            bboxFrame = self.createBoundingBox(frame, threshold)
            video_writer.write(bboxFrame)
            cv2.imshow('RTSP Stream Detection', bboxFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    def showImagePopup(self, image):
        # Create a GUI window
        root1 = tk.Tk()
        root1.title("Image Detection Result")

        # Convert the image to a format that Tkinter can display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        # Create a label to display the image
        label = tk.Label(root1, image=photo)
        label.pack()

        # Close the GUI window when the user clicks anywhere on the image
        label.bind("<Button-1>", lambda e: root1.destroy())

        # Run the GUI event loop
        root1.mainloop()

def main():
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    detector = Detector(modelURL, classFile)

    root = tk.Tk()
    app = Interface(root, detector)
    root.mainloop()

if __name__ == "__main__":
    main()
