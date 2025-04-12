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
from datetime import datetime
import csv

np.random.seed(20)

class Detector:
    def __init__(self, modelURL, classFile):
        self.model = None
        self.classesList = None
        self.colorList = None
        self.modelName = None
        self.cacheDir = None
        self.max_people_count = 0
        self.current_people_count=0
        self.max_time = current_time = datetime.now().strftime("%I:%M:%S %p")
        self.downloadModel(modelURL)
        self.loadModel()
        self.readClasses(classFile)
        self.detection_ids = {
            "person": 1,
            "motorcycle": 1,
            "bicycle": 1,
            "car": 1,
            "bus": 1,
            "truck": 1,
        }

    def get_detection_stats(self):
        return {
            "max_people_count": self.max_people_count,
            "current_people_count": 0,  # Update this dynamically during real-time processing if needed
            "max_time": self.max_time
        }

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.tar.gz')]

        self.cacheDir = "./pretrained_models"
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
        current_people_count = 0  
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold=threshold, score_threshold=threshold)

        # Define classes to log
        log_classes = {"person", "motorcycle", "bicycle", "car", "bus", "truck"}
        csv_file = "detection_log2.csv"

        # Ensure CSV file has a header
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "Class Name", "Date", "Time (IST)", 
                                 "Dimensions (Width x Height)", 
                                 "Xmin", "Ymin", "Xmax", "Ymax", 
                                 "Frame Width", "Frame Height","current_people_count","max_people_count","max_time"])
        # Get current date and time in IST
        current_date = datetime.now().strftime("%d-%m-%Y")
        current_time = datetime.now().strftime("%I:%M:%S %p")
        if(current_time=="00:00:00 AM"):
            self.max_people_count=0
            self.max_time="00:00:00 AM"
          # To maintain unique ID for detections
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex].lower()

                if classLabelText not in log_classes:
                    continue
                if classLabelText == "person":
                    current_people_count += 1  # Increment person count

                classColor = self.colorList[classIndex]
                displayText = '{}: {}%'.format(classLabelText.upper(), classConfidence)

                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)

                obj_width = xmax - xmin
                obj_height = ymax - ymin
                dimensions = f"{obj_width}x{obj_height}"

                

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                if self.max_people_count < current_people_count:
                    self.max_people_count=current_people_count
                    self.max_time=current_time
                    
                # Log details to CSV
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.detection_ids[classLabelText], classLabelText, current_date, current_time, 
                                     dimensions, xmin, ymin, xmax, ymax, imW, imH,current_people_count,self.max_people_count,self.max_time])

                self.detection_ids[classLabelText] += 1  # Increment ID for next detection

                ########################################
                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))

                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
        
        self.current_people_count=current_people_count
        # print("--- ////    \\\\ -----",self.max_people_count,"  ",current_people_count)
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

    def predictVideoSource(self, videoPath, threshold=0.5, save_path=None, process_every_nth_frame=35):
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        print("Output saved at:", output_path)

    def predictRTSPStream(self, channel, threshold=0.5, process_every_nth_frame=35):
        ch = str(channel)
        rtspStreamURL = "rtsp://admin:DK@admin85@172.31.37.125:554/cam/realmonitor?channel=" + ch + "&subtype=0"
        cap = cv2.VideoCapture(rtspStreamURL)

        if not cap.isOpened():
            print("Error opening video stream or file")
            return

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

            cv2.imshow('RTSP Stream Detection', bboxFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def showImagePopup(self, image):
        # Create a GUI window
        root1 = tk.Tk()
        root1.title("Image Detection Result")

        # Convert the image to a format that Tkinter can display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        # Create a label to display the
        label = tk.Label(root1, image=photo)
        label.image = photo
        label.pack()

        # Button to close the window
        close_button = tk.Button(root1, text="Close", command=root1.quit)
        close_button.pack()

        root1.mainloop()

# Main function to run the detector
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection using a pre-trained model.")
    parser.add_argument('--model', type=str, required=True, help="URL of the pre-trained model")
    parser.add_argument('--classes', type=str, required=True, help="Path to the class labels file")
    parser.add_argument('--image', type=str, help="Path to the image file for detection")
    parser.add_argument('--video', type=str, help="Path to the video file for detection")
    parser.add_argument('--stream', type=int, help="RTSP stream channel for detection")
    parser.add_argument('--threshold', type=float, default=0.5, help="Detection threshold")

    args = parser.parse_args()

    # Initialize the detector with model and classes
    detector = Detector(args.model, args.classes)

    # Choose the type of detection based on user input
    if args.image:
        detector.predictImage(args.image, threshold=args.threshold)
    elif args.video:
        detector.predictVideoSource(args.video, threshold=args.threshold)
    elif args.stream:
        detector.predictRTSPStream(args.stream, threshold=args.threshold)
    else:
        print("Please provide either an image, video, or RTSP stream channel for detection.")
