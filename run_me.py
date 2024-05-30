import argparse
from Detector import Detector

parser = argparse.ArgumentParser(description='Object Detection using TensorFlow')
parser.add_argument('--image', type=str, help='Path to the image file for processing')

args = parser.parse_args()

if args.image:
    imagePath = args.image
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    threshold = 0.5

    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()

    detector.predictImage(imagePath, threshold)
else:
    print("Please provide an image file path using the --image argument.")
