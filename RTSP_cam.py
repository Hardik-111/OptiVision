import argparse
# from Detector import Detector
import cv2
parser = argparse.ArgumentParser(description='Object Detection using TensorFlow')
parser.add_argument('--channel', type=int, help='Path to the image file for processing')

args = parser.parse_args()

if args.channel:
    # Function to resize frames
    def resize_frame(frame, width=None, height=None):
        if width is not None and height is not None:
            return cv2.resize(frame, (width, height))
        elif width is not None:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            return cv2.resize(frame, (width, int(width / aspect_ratio)))
        elif height is not None:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            return cv2.resize(frame, (int(height * aspect_ratio), height))
        else:
            return frame

    # RTSP stream URLs for multiple cameras
    ch=str(args.channel)
    url='rtsp://admin:DK@admin85@172.31.37.125:554/cam/realmonitor?channel='+ch+'&subtype=0'
    print(url)
    rtsp_urls = [url]

    # Open video capture objects for each camera
    caps = [cv2.VideoCapture(url) for url in rtsp_urls]

    # Check if the streams are opened successfully
    for i, cap in enumerate(caps):
        print(i,"    ",cap)
        if not cap.isOpened():
            #print(f"Error: Could not open the RTSP stream for camera {i+1}.")
            print("Error: Could not open the RTSP stream for camera {}.".format(i+1)) 
            exit()

    # Set the desired size for displayed frames
    frame_width = 640
    frame_height = 480

    # Read and display frames from each camera
    while True:
        # print(caps[0].read()[1])
        frames = [cap.read()[1] for cap in caps]
        resized_frames = [resize_frame(frame, frame_width, frame_height) for frame in frames]
        
        for i, frame in enumerate(resized_frames):
            # cv2.imshow('Camera ' + str(i+1), frame)
            cv2.imshow(f'Camera {i+1}', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture objects
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

else:
    print("Please provide an channel path using the --image argument.")