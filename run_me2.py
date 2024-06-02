import argparse
from Detector import Detector
parser = argparse.ArgumentParser(description='Object Detection using TensorFlow')
parser.add_argument('--channel', type=int, help='Path to the image file for processing')

args = parser.parse_args()
# Parameters
if args.channel:
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    ch=str(args.channel)
    rtspStreamURL = "rtsp://admin:DK@admin85@172.31.37.125:554/cam/realmonitor?channel="+ch+"&subtype=0"
    threshold = 0.5

    # Initialize detector
    detector = Detector()

    # Load classes and download model
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()


    # Predict on an RTSP stream
    detector.predictRTSPStream(rtspStreamURL, threshold)
