import argparse
from Detector import Detector

parser = argparse.ArgumentParser(description='Object Detection using TensorFlow')
parser.add_argument('--video', type=str, help='Path to the video file for processing')

args = parser.parse_args()

if args.video:
    videoPath = args.video
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    threshold = 0.5

    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()

    detector.predictVideoSource(videoPath, threshold)
else:
    print("Please provide a video file path using the --video argument.")
