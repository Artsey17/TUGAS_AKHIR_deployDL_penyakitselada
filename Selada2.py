import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import time
import yaml
import os
import sys
import argparse
import glob
try:
    import ncnn
except ImportError:
    print("ERROR: NCNN not installed. Install NCNN and its Python bindings.")
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
        self.net = None
        self.labels = None
        self.bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
                           (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]
        self.min_thresh = 0.5
        self.resolution = "640x480"
        self.camera_index = 0
        self.use_picamera = False
        self.record = False
        self.recorder = None
        self.input_size = (320, 320)  # Sesuai metadata NCNN Anda

        # Parse argumen baris perintah jika ada
        self.parse_args(model_path, source_path)

        # Inisialisasi model jika path diberikan
        if self.model_path:
            self.load_model(self.model_path)

        if root:
            self.setup_gui()

    def parse_args(self, model_path, source_path):
        parser = argparse.ArgumentParser(description="YOLO Object Detection with NCNN")
        parser.add_argument('--model', type=str, help='Path to NCNN model file or folder (example: "SGD.0.005_ncnn_model")',
                            required=False)
        parser.add_argument('--source', type=str, help='Image source: file ("test.jpg"), folder ("test_dir"), video ("testvid.mp4"), usb0, or picamera0',
                            required=False)
        parser.add_argument('--thresh', type=float, help='Minimum confidence threshold (example: "0.4")',
                            default=0.5)
        parser.add_argument('--resolution', type=str, help='Resolution in WxH (example: "640x480")',
                            default="640x480")
        parser.add_argument('--record', action='store_true', help='Record results as "demo1.avi". Requires --resolution.')
        args = parser.parse_args()

        if args.model:
            self.model_path = args.model
        elif model_path:
            self.model_path = model_path
        else:
            self.model_path = "SGD.0.005_ncnn_model"  # Default path NCNN

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
        param_path = model_path if model_path.endswith(".param") else os.path.join(model_path, "best.param")
        bin_path = os.path.splitext(param_path)[0] + ".bin"
        metadata_path = os.path.join(os.path.dirname(param_path), "metadata.yaml")

        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            print(f'ERROR: Model NCNN not found at: {param_path} or {bin_path}')
            if self.root:
                messagebox.showerror("Error", f"Model NCNN not found at: {param_path} or {bin_path}")
            sys.exit(0)

        try:
            self.net = ncnn.Net()
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as file:
                    metadata = yaml.safe_load(file)
                self.labels = list(metadata["names"].values())
                self.input_size = tuple(metadata["imgsz"])  # Misalnya, [320, 320]
            print(f"Kelas kustom: {self.labels}")
        except Exception as e:
            print(f"ERROR: Failed to load NCNN model: {str(e)}")
            if self.root:
                messagebox.showerror("Error", f"Failed to load NCNN model: {str(e)}")
            sys.exit(1)

    def setup_gui(self):
        self.root.title("YOLO Object Detection (NCNN)")
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

    def process_yolo_output(self, outputs, frame):
        boxes = []
        confidences = []
        class_ids = []
        h, w = frame.shape[:2]

        for detection in outputs:
            scores = detection[4:]  # Asumsikan [x, y, w, h, class_scores...]
            confidence = np.max(scores)
            class_id = np.argmax(scores)
            if confidence > self.min_thresh and class_id < len(self.labels):
                center_x = detection[0] * w
                center_y = detection[1] * h
                width = detection[2] * w
                height = detection[3] * h
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, x + width, y + height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.min_thresh, 0.5)
        object_count = len(indices)
        for i in indices:
            box = boxes[i]
            x, y, x2, y2 = box
            color = self.bbox_colors[class_ids[i] % 10]
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            label = f'{self.labels[class_ids[i]]}: {int(confidence*100)}%'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(y, label_size[1] + 10)
            cv2.rectangle(frame, (x, label_ymin - label_size[1] - 10), (x + label_size[0], label_ymin + 10), color, cv2.FILLED)
            cv2.putText(frame, label, (x, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame, object_count

    def process_source(self, source, source_type):
        self.is_processing = True
        resW, resH = map(int, self.resolution.split("x"))
        resize = True

        # Validasi perekaman
        if self.record:
            if source_type not in ["video", "usb", "picamera"]:
                print('Recording only works for video and camera sources.')
                if self.root:
                    messagebox.showerror("Error", "Recording only works for video and camera sources.")
                self.is_processing = False
                return
            self.recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

        # Inisialisasi sumber
        img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
        vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
        imgs_list = []

        if source_type in ["image", "video"]:
            if os.path.isfile(source):
                _, ext = os.path.splitext(source)
                if ext in img_ext_list:
                    imgs_list = [source]
                    source_type = "image"
                elif ext in vid_ext_list:
                    source_type = "video"
                else:
                    print(f'File extension {ext} is not supported.')
                    if self.root:
                        messagebox.showerror("Error", f"File extension {ext} is not supported.")
                    self.is_processing = False
                    return
            else:
                print(f'File {source} not found.')
                if self.root:
                    messagebox.showerror("Error", f"File {source} not found.")
                self.is_processing = False
                return
        elif source_type == "folder":
            if os.path.isdir(source):
                filelist = glob.glob(source + '/*')
                for file in filelist:
                    _, file_ext = os.path.splitext(file)
                    if file_ext in img_ext_list:
                        imgs_list.append(file)
            else:
                print(f'Directory {source} not found.')
                if self.root:
                    messagebox.showerror("Error", f"Directory {source} not found.")
                self.is_processing = False
                return
        elif source_type in ["usb", "picamera"]:
            if source_type == "usb":
                cap = cv2.VideoCapture(source)  # Gunakan indeks langsung
                if not cap.isOpened():
                    print(f"ERROR: Unable to open USB camera at index {source}.")
                    if self.root:
                        messagebox.showerror("Error", f"Unable to open USB camera at index {source}.")
                    self.is_processing = False
                    return
                if resize:
                    cap.set(3, resW)
                    cap.set(4, resH)
            elif source_type == "picamera" and Picamera2:
                cap = Picamera2()
                cap.configure(cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)}))
                cap.start()
            else:
                print("PiCamera not supported or not installed.")
                if self.root:
                    messagebox.showerror("Error", "PiCamera not supported or not installed.")
                self.is_processing = False
                return

        frame_rate_buffer = []
        fps_avg_len = 200
        img_count = 0
        avg_frame_rate = 0.0  # Inisialisasi dengan nilai default

        while self.is_processing:
            t_start = time.perf_counter()

            # Load frame
            if source_type in ["image", "folder"]:
                if img_count >= len(imgs_list):
                    print('All images have been processed. Exiting.')
                    break
                img_filename = imgs_list[img_count]
                frame = cv2.imread(img_filename)
                if frame is None:
                    print(f"ERROR: Cannot load image: {img_filename}")
                    img_count += 1
                    continue
                if resize:
                    frame = cv2.resize(frame, (resW, resH))
                img_count += 1
            elif source_type in ["video", "usb", "picamera"]:
                if source_type == "video":
                    ret, frame = cap.read()
                    if not ret:
                        print('Reached end of video or camera disconnected.')
                        break
                elif source_type == "usb":
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print('Unable to read frames from USB camera.')
                        break
                elif source_type == "picamera":
                    frame = cap.capture_array()
                    if frame is None:
                        print('Unable to read frames from PiCamera.')
                        break
                if resize:
                    frame = cv2.resize(frame, (resW, resH))

            # Preprocessing untuk NCNN
            blob = cv2.resize(frame, self.input_size)
            blob = blob.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)  # HWC ke CHW
            blob = np.ascontiguousarray(blob, dtype=np.float32)

            # Inferensi NCNN
            try:
                ex = self.net.create_extractor()
                ex.input("data", ncnn.Mat(blob.shape[1], blob.shape[2], blob.shape[0], (blob.ctypes.data_as(ncnn.Mat.data_t))))
                ret, out = ex.extract("output")  # Sesuaikan nama layer output
                outputs = np.array(out).reshape(-1, 4 + len(self.labels))
                print(f"Output shape: {outputs.shape}")
            except Exception as e:
                print(f"ERROR: Failed to run NCNN inference: {str(e)}")
                if self.root:
                    messagebox.showerror("Error", f"Failed to run NCNN inference: {str(e)}")
                break

            # Proses output
            frame, object_count = self.process_yolo_output(outputs, frame)

            # Display FPS and object count
            if source_type in ["video", "usb", "picamera"] and frame_rate_buffer:
                cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('YOLO Detection (NCNN)', frame)
            if self.record and self.recorder:
                self.recorder.write(frame)

            # Handle keyboard input
            key = cv2.waitKey(0 if source_type in ["image", "folder"] else 5)
            if key == ord('q') or key == ord('Q'):
                self.is_processing = False
            elif key == ord('s') or key == ord('S'):
                cv2.waitKey(0)
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite('capture.png', frame)

            # Calculate FPS
            t_stop = time.perf_counter()
            frame_rate_calc = 1 / (t_stop - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = np.mean(frame_rate_buffer) if frame_rate_buffer else 0.0

        # Clean up
        print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
        if source_type in ["video", "usb"]:
            cap.release()
        elif source_type == "picamera":
            cap.stop()
        if self.record and self.recorder:
            self.recorder.release()
        cv2.destroyAllWindows()
        self.is_processing = False

    def quit_app(self):
        if not self.is_processing:
            self.root.quit()

def command_line_mode(model_path=None, source_path=None):
    app = YOLOApp(root=None, model_path=model_path, source_path=source_path)
    if source_path:
        if os.path.isdir(source_path):
            app.process_source(source_path, source_type="folder")
        elif os.path.isfile(source_path):
            _, ext = os.path.splitext(source_path)
            if ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
                app.process_source(source_path, source_type="image")
            elif ext in ['.avi', '.mov', '.mp4', '.mkv', '.wmv']:
                app.process_source(source_path, source_type="video")
            else:
                print(f'File extension {ext} is not supported.')
                sys.exit(0)
        elif 'usb' in source_path:
            app.process_source(int(source_path[3:]), source_type="usb")
        elif 'picamera' in source_path:
            app.process_source("picamera0", source_type="picamera")
        else:
            print(f'Input {source_path} is invalid.')
            sys.exit(0)
    else:
        if Picamera2:
            use_picamera = input("Gunakan PiCamera? (y/n, default n): ").lower() == "y"
            if use_picamera:
                app.process_source("picamera0", source_type="picamera")
                return
        app.process_source(0, source_type="usb")  # Default ke indeks 0

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = YOLOApp(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"ERROR: Failed to start GUI: {str(e)}")
        print("Switching to command-line mode...")
        command_line_mode()
