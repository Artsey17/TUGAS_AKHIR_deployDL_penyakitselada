import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import time
import os
import sys
import argparse
import glob
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip3 install ultralytics")
    sys.exit(1)
try:
    from PIL import Image, ImageTk
except ImportError:
    print("WARNING: Pillow not installed. Install with: pip3 install pillow")
    ImageTk = None
try:
    from picamera2 import Picamera2
except ImportError:
    print("WARNING: picamera2 not installed. PiCamera support disabled.")
    Picamera2 = None

class YOLOApp:
    def __init__(self, root=None, model_path=None, source_path=None):
        self.root = root
        self.is_processing = False
        self.model = None
        self.labels = None
        self.bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
                           (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]
        self.min_thresh = 0.5
        self.resolution = "640x480"
        self.camera_index = 0
        self.use_picamera = False
        self.record = False
        self.recorder = None

        # Parse argumen baris perintah jika ada
        self.parse_args(model_path, source_path)

        # Inisialisasi model jika path diberikan
        if self.model_path:
            self.load_model(self.model_path)

        if root:
            self.setup_gui()

    def parse_args(self, model_path, source_path):
        parser = argparse.ArgumentParser(description="YOLO Object Detection")
        parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                            required=False)
        parser.add_argument('--source', help='Image source: file ("test.jpg"), folder ("test_dir"), video ("testvid.mp4"), usb0, or picamera0',
                            required=False)
        parser.add_argument('--thresh', type=float, help='Minimum confidence threshold (example: "0.4")',
                            default=0.5)
        parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480")',
                            default="640x480")
        parser.add_argument('--record', action='store_true', help='Record results as "demo1.avi". Requires --resolution.')
        args = parser.parse_args()

        if args.model:
            self.model_path = args.model
        elif model_path:
            self.model_path = model_path
        else:
            self.model_path = None

        if args.source:
            self.source_path = args.source
        elif source_path:
            self.source_path = source_path
        else:
            self.source_path = None

        self.min_thresh = args.thresh if args.thresh else 0.5
        self.resolution = args.resolution if args.resolution else "640x480"
        self.record = args.record

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f'ERROR: Model path is invalid or model was not found: {model_path}')
            if self.root:
                messagebox.showerror("Error", f"Model path is invalid or model was not found: {model_path}")
            sys.exit(0)

        try:
            self.model = YOLO(model_path, task='detect')
            self.labels = self.model.names
            print(f"Kelas kustom: {self.labels}")
        except Exception as e:
            print(f"ERROR: Gagal memuat model: {str(e)}")
            if self.root:
                messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")
            sys.exit(1)

    def setup_gui(self):
        self.root.title("YOLO Object Detection")
        self.root.geometry("1024x600")

        title_frame = tk.Frame(self.root)
        title_frame.pack(fill="x", padx=10, pady=10)
        
        if ImageTk:
            try:
                logo_path = "/home/pi/project/Logo_UGM.png"
                if os.path.exists(logo_path):
                    image = Image.open(logo_path).resize((50, 50))
                    self.logo = ImageTk.PhotoImage(image)
                    tk.Label(title_frame, image=self.logo).pack(side="left", padx=5)
                    tk.Frame(title_frame, width=5, bg="black").pack(side="left", padx=5)
                else:
                    print(f"WARNING: Logo UGM not found at {logo_path}. Skipping logo.")
            except Exception as e:
                messagebox.showwarning("Peringatan", f"Gagal memuat logo: {str(e)}")

        text_frame = tk.Frame(title_frame)
        text_frame.pack(side="left", fill="x", expand=True)
        tk.Label(text_frame, text="PENERAPAN DEEP LEARNING YOLO BERDASARKAN CITRA DIGITAL UNTUK MENDETEKSI PENYAKIT DAUN SELADA: PERFORMA MODEL YOLOV11N DAN YOLOV12N BERBASIS RASPBERRY PI", font=("Arial", 14, "bold"), wraplength=950, justify="center").pack(pady=5)
        tk.Label(text_frame, text="Theophylus Yestra Pratama, 21/479277/SV/19471", font=("Arial", 12), justify="center").pack(pady=5)
        tk.Label(text_frame, text="Deteksi Penyakit Tanaman Selada", font=("Arial", 12, "italic"), justify="center").pack(pady=5)
        tk.Frame(title_frame, height=2, bg="black").pack(fill="x", pady=5)

        tk.Button(self.root, text="Upload Gambar", command=self.upload_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)
        tk.Button(self.root, text="Upload Video", command=self.upload_video, font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5).pack(pady=10)
        tk.Button(self.root, text="Real-Time Streaming", command=self.start_streaming, font=("Arial", 12), bg="#FFC107", fg="black", padx=10, pady=5).pack(pady=10)
        
        tk.Label(self.root, text="Resolusi:", font=("Arial", 10)).pack(pady=5)
        self.resolution_var = tk.StringVar(value=self.resolution)
        resolutions = ["640x480", "800x600", "1024x600", "1280x720"]
        ttk.Combobox(self.root, textvariable=self.resolution_var, values=resolutions, state="readonly", font=("Arial", 10)).pack(pady=5)
        
        tk.Label(self.root, text="Kamera:", font=("Arial", 10)).pack(pady=5)
        self.camera_var = tk.StringVar(value="0")  # Default ke indeks 0
        cameras = ["0", "1", "2", "picamera0"] if Picamera2 else ["0", "1", "2"]
        ttk.Combobox(self.root, textvariable=self.camera_var, values=cameras, state="readonly", font=("Arial", 10)).pack(pady=5)
        
        tk.Button(self.root, text="Keluar", command=self.quit_app, font=("Arial", 12), bg="#F44336", fg="white", padx=10, pady=5).pack(pady=10)

    def upload_image(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if file:
            self.resolution = self.resolution_var.get()
            self.process_source(file, source_type="image")

    def upload_video(self):
        file = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if file:
            self.resolution = self.resolution_var.get()
            self.process_source(file, source_type="video")

    def start_streaming(self):
        self.resolution = self.resolution_var.get()
        camera = self.camera_var.get()
        if camera == "picamera0" and Picamera2:
            self.use_picamera = True
            self.process_source("picamera0", source_type="picamera")
        else:
            self.use_picamera = False
            self.camera_index = int(camera)
            self.process_source(self.camera_index, source_type="usb")

    def process_source(self, source, source_type):
        self.is_processing = True
        resize = False
        user_res = self.resolution
        if user_res:
            resize = True
            resW, resH = map(int, user_res.split('x'))
        else:
            resW, resH = 640, 480  # Default resolution if not specified

        # Check if recording is valid and set up recording
        recorder = None
        if self.record:
            if source_type not in ['video', 'usb', 'picamera']:
                print('Recording only works for video and camera sources. Please try again.')
                if self.root:
                    messagebox.showerror("Error", "Recording only works for video and camera sources.")
                self.is_processing = False
                return
            if not user_res:
                print('Please specify resolution to record video at.')
                if self.root:
                    messagebox.showerror("Error", "Please specify resolution to record video at.")
                self.is_processing = False
                return
            recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

        # Load or initialize image source
        img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
        vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
        imgs_list = []

        if source_type == 'image':
            imgs_list = [source]
        elif source_type == 'folder':
            filelist = glob.glob(source + '/*')
            for file in filelist:
                _, file_ext = os.path.splitext(file)
                if file_ext in img_ext_list:
                    imgs_list.append(file)
        elif source_type in ['video', 'usb', 'picamera']:
            if source_type == 'video':
                cap_arg = source
            elif source_type == 'usb':
                cap_arg = source  # Gunakan indeks langsung
            elif source_type == 'picamera':
                cap_arg = source
            cap = cv2.VideoCapture(cap_arg) if source_type != 'picamera' else Picamera2()
            if source_type != 'picamera':
                if not cap.isOpened():
                    print(f"ERROR: Unable to open video/camera source (index {cap_arg}).")
                    if self.root:
                        messagebox.showerror("Error", f"Unable to open camera at index {cap_arg}.")
                    self.is_processing = False
                    return
                if resize:
                    cap.set(3, resW)
                    cap.set(4, resH)
            else:
                cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
                cap.start()

        # Initialize control and status variables
        avg_frame_rate = 0
        frame_rate_buffer = []
        fps_avg_len = 200
        img_count = 0

        # Begin inference loop
        while self.is_processing:
            t_start = time.perf_counter()

            # Load frame from image source
            if source_type in ['image', 'folder']:
                if img_count >= len(imgs_list):
                    print('All images have been processed. Exiting program.')
                    break
                img_filename = imgs_list[img_count]
                frame = cv2.imread(img_filename)
                img_count += 1
                if frame is None:
                    print(f"ERROR: Cannot load image: {img_filename}")
                    continue
            elif source_type in ['video', 'usb', 'picamera']:
                if source_type == 'video':
                    ret, frame = cap.read()
                    if not ret:
                        print('Reached end of the video file or camera disconnected. Exiting program.')
                        break
                elif source_type == 'usb':
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print('Unable to read frames from the camera. Exiting program.')
                        break
                elif source_type == 'picamera':
                    frame = cap.capture_array()
                    if frame is None:
                        print('Unable to read frames from the PiCamera. Exiting program.')
                        break
            else:
                continue

            # Resize frame to desired display resolution
            if resize:
                frame = cv2.resize(frame, (resW, resH))

            # Run inference on frame
            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            # Go through each detection and get bbox coords, confidence, and class
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                conf = detections[i].conf.item()

                if conf > float(self.min_thresh):
                    color = self.bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f'{self.labels[classidx]}: {int(conf*100)}%'
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    object_count += 1

            # Calculate and draw framerate
            if source_type in ['video', 'usb', 'picamera']:
                cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display detection results
            cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('YOLO detection results', frame)
            if self.record and recorder:
                recorder.write(frame)

            # Handle keyboard input
            key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
            if key == ord('q') or key == ord('Q'):
                self.is_processing = False
            elif key == ord('s') or key == ord('S'):
                cv2.waitKey(0)
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite('capture.png', frame)

            # Calculate FPS for this frame
            t_stop = time.perf_counter()
            frame_rate_calc = 1 / (t_stop - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = np.mean(frame_rate_buffer) if frame_rate_buffer else 0.0

        # Clean up
        print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
        if source_type in ['video', 'usb']:
            cap.release()
        elif source_type == 'picamera':
            cap.stop()
        if self.record and recorder:
            recorder.release()
        cv2.destroyAllWindows()
        self.is_processing = False

    def quit_app(self):
        if not self.is_processing:
            self.root.quit()

def command_line_mode(model_path=None, source_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=True)
    parser.add_argument('--source', help='Image source', required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5)
    parser.add_argument('--resolution', help='Resolution in WxH', default=None)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    if not os.path.exists(model_path):
        print('ERROR: Model path is invalid or model was not found.')
        sys.exit(0)

    model = YOLO(model_path, task='detect')
    labels = model.names

    img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
    vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = 'image'
        elif ext in vid_ext_list:
            source_type = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(0)
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])
    elif 'picamera' in img_source:
        source_type = 'picamera'
        picam_idx = int(img_source[8:])
    else:
        print(f'Input {img_source} is invalid.')
        sys.exit(0)

    resize = False
    if user_res:
        resize = True
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

    if record:
        if source_type not in ['video', 'usb']:
            print('Recording only works for video and camera sources.')
            sys.exit(0)
        if not user_res:
            print('Please specify resolution to record video at.')
            sys.exit(0)
        recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))
    else:
        recorder = None

    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        imgs_list = []
        filelist = glob.glob(img_source + '/*')
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)
    elif source_type == 'video' or source_type == 'usb':
        if source_type == 'video':
            cap_arg = img_source
        elif source_type == 'usb':
            cap_arg = usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if user_res:
            cap.set(3, resW)
            cap.set(4, resH)
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
        cap.start()

    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    while True:
        t_start = time.perf_counter()

        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        elif source_type == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None:
                print('Unable to read frames from the camera. Exiting program.')
                break
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is None:
                print('Unable to read frames from the PiCamera. Exiting program.')
                break

        if resize and frame is not None:
            frame = cv2.resize(frame, (resW, resH))

        if frame is not None:
            results = model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                conf = detections[i].conf.item()

                if conf > float(min_thresh):
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f'{labels[classidx]}: {int(conf*100)}%'
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    object_count += 1

            if source_type in ['video', 'usb', 'picamera']:
                cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('YOLO detection results', frame)
            if record and recorder:
                recorder.write(frame)

            key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                cv2.waitKey(0)
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite('capture.png', frame)

            t_stop = time.perf_counter()
            frame_rate_calc = 1 / (t_stop - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = np.mean(frame_rate_buffer) if frame_rate_buffer else 0.0

    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record and recorder:
        recorder.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = YOLOApp(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"ERROR: Failed to start GUI: {str(e)}")
        print("Switching to command-line mode...")
        command_line_mode()
