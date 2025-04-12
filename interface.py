# interface.py

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import subprocess
from Detector import Detector

class Interface:
    def __init__(self, master, detector):
        self.master = master
        self.master.title("Object Detection Using TensorFlow")
        self.master.geometry("600x400")
        self.detector = detector  # Pass the detector object

        # Load background image (handle potential errors)
        try:
            bg_image = Image.open("chatbot-bcg.jpg")  # Assuming background image filename is "bcg.jpg"
            self.background_photo = ImageTk.PhotoImage(bg_image)
            self.bg_label = tk.Label(master, image=self.background_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except FileNotFoundError:
            messagebox.showerror("Error", "Background image 'bcg.jpg' not found. Using default background.")
            # Consider using a default background image here (optional)

        # Heading Label
        heading_label = tk.Label(master, text="Object Detection", font=("Helvetica", 20, "bold"), bg="white")
        heading_label.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

        # Image Processing Section
        image_frame = tk.Frame(master, bg="white", bd=5)
        image_frame.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

        image_button = self.create_button(image_frame, "Process Image", self.process_image, 0, 0)
        video_button = self.create_button(image_frame, "Process Video", self.process_video, 0, 1)
        rtsp_button = self.create_button(image_frame, "Process RTSP Stream", self.process_rtsp, 0, 2)
        drtspStream_button = self.create_button(image_frame, "Choose channel", self.rtsp_stream, 0, 3)

    def create_button(self, frame, text, command, row, column):
        button = tk.Button(frame, text=text, font=("Helvetica", 12), command=command)
        button.grid(row=row, column=column, padx=10, pady=10)
        return button

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.detector.predictImage(file_path)

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.detector.predictVideoSource(file_path)

    def process_rtsp(self):
        channel = simpledialog.askstring("RTSP Stream URL Processing ", "Enter channel no.:")
        if channel:
            self.detector.predictRTSPStream(channel)

    def rtsp_stream(self):
        channel = simpledialog.askstring("RTSP Stream URL", "Enter channel no.:")
        if channel:
            subprocess.run(["python", "RTSP_cam.py", "--channel", channel])

    def show_output(self, output_file_path):
        try:
            output_image = Image.open(output_file_path)
            output_image = output_image.resize((300, 200), resample=Image.ANTIALIAS)
            output_photo = ImageTk.PhotoImage(output_image)

            output_label = tk.Label(self.master, image=output_photo)
            output_label.image = output_photo
            output_label.pack(pady=10)
        except FileNotFoundError:
            messagebox.showerror("Error", "Output file not found.")

def main():
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz"
    classFile = "coco.names"

    root = tk.Tk()
    detector = Detector(modelURL, classFile)
    gui = Interface(root, detector)
    root.mainloop()

if __name__ == "__main__":
    main()









