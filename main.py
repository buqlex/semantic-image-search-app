import requests
import ultralytics
import os
import sys
import pathlib  # For redirecting PosixPath
import torch  # For loading the YOLOv5 model
import cv2  # For image processing
from pathlib import Path  # For working with paths
from tkinterdnd2 import DND_FILES, TkinterDnD  # Import Drag-and-Drop support
import tkinter as tk
from tkinter import filedialog, Label, Frame, messagebox
from tkinter import ttk  # For Progressbar
from PIL import Image, ImageTk  # For resizing image
import platform  # For detecting the operating system
import warnings  # For filtering warnings
import threading  # For running video processing in a separate thread
import queue  # For thread-safe communication
import multiprocessing  # For setting number of threads
import logging  # For logging events
import csv  # For exporting detection data
import time  # For measuring processing time


class AerialObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecting Aerial Objects")
        self.root.geometry("1200x900")  # Increased width and height for additional controls

        # Initialize logging
        logging.basicConfig(
            filename='app.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logging.info("Application started.")

        # Redirecting PosixPath to WindowsPath (for compatibility on Windows systems)
        self.temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        # Specify the path to your models
        self.MODEL_FILES = {
            "Small Model": "bests.pt",
            "Medium Model": "bestm.pt",
            "Large Model": "best.pt"
        }

        # Set default model
        self.selected_model_name = "Large Model"
        self.MODEL_PATH = self.MODEL_FILES[self.selected_model_name]

        # Default save directory
        self.save_directory = Path("result")
        self.save_directory.mkdir(exist_ok=True)  # Create the directory if it does not exist

        # Suppress all FutureWarnings to prevent console flooding
        warnings.simplefilter("ignore", FutureWarning)

        # Set number of threads to number of CPU cores for PyTorch
        num_threads = multiprocessing.cpu_count()
        torch.set_num_threads(num_threads)
        logging.info(f"Set PyTorch to use {num_threads} threads.")

        # Create a queue for progress updates
        self.progress_queue = queue.Queue()

        # Create an event for cancellation
        self.cancel_event = threading.Event()

        # Load the YOLOv5 model
        self.load_model()

        # Initialize GUI components
        self.create_menu()
        self.create_widgets()

        # Start the progress update loop
        self.root.after(100, self.update_progress)

    def load_model(self):
        try:
            # Определяем базовый путь в зависимости от того, из .exe запускается или нет
            if getattr(sys, 'frozen', False):
                base_path = Path(sys._MEIPASS)
            else:
                base_path = Path(__file__).parent

            # Путь к папке yolov5, добавленной при упаковке
            yolov5_path = base_path / 'yolov5'

            # Load the selected YOLOv5 model
            self.model = torch.hub.load(str(yolov5_path), "custom", path=self.MODEL_PATH, source="local",
                                        force_reload=True)
            self.model.to('cpu')  # Move model to CPU
            self.model.eval()  # Set model to evaluation mode
            logging.info(f"YOLOv5 model '{self.selected_model_name}' successfully loaded from {self.MODEL_PATH}.")
        except Exception as e:
            logging.error(f"Error loading the YOLOv5 model '{self.selected_model_name}': {e}")
            messagebox.showerror("Model Loading Error",
                                 f"Error loading the YOLOv5 model '{self.selected_model_name}':\n{e}")
            pathlib.PosixPath = self.temp
            self.root.destroy()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        helpmenu.add_separator()
        helpmenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="Menu", menu=helpmenu)
        self.root.config(menu=menubar)

    def show_about(self):
        messagebox.showinfo("About", "Detecting Aerial Objects\nVersion 1.0\n© 2024 Kolomiiets D.V")

    def create_widgets(self):
        # Drop Frame
        drop_frame = Frame(self.root, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=2)
        drop_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=False)

        drop_label = Label(
            drop_frame,
            text="Drop an image or video file here or click 'Upload File' below",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#333",
            wraplength=1100,
            justify="center",
        )
        drop_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind("<<Drop>>", self.handle_drop)

        # Upload Button
        upload_button = tk.Button(
            self.root,
            text="Upload File",
            command=self.upload_file,
            font=("Arial", 14),
            bg="lightblue",
            fg="black",
        )
        upload_button.pack(pady=10)

        # Cancel Button
        self.cancel_button = tk.Button(
            self.root,
            text="Cancel",
            command=self.cancel_processing,
            font=("Arial", 14),
            bg="red",
            fg="white",
            state='disabled'  # Initially disabled
        )
        self.cancel_button.pack(pady=5)

        # Settings Frame
        settings_frame = Frame(self.root)
        settings_frame.pack(padx=20, pady=10, fill=tk.X)

        # Resolution Setting
        resolution_label = Label(settings_frame, text="Frame Resolution:", font=("Arial", 12))
        resolution_label.pack(side=tk.LEFT, padx=5)

        self.resolution_var = tk.StringVar(value="640x480")
        resolution_options = ["320x240", "640x480", "800x600", "1280x720"]
        resolution_menu = ttk.Combobox(settings_frame, textvariable=self.resolution_var, values=resolution_options,
                                       state="readonly")
        resolution_menu.pack(side=tk.LEFT, padx=5)

        # Model Selection
        model_label = Label(settings_frame, text="Select Model:", font=("Arial", 12))
        model_label.pack(side=tk.LEFT, padx=20)

        self.model_var = tk.StringVar(value=self.selected_model_name)
        model_options = list(self.MODEL_FILES.keys())
        model_menu = ttk.Combobox(settings_frame, textvariable=self.model_var, values=model_options, state="readonly")
        model_menu.pack(side=tk.LEFT, padx=5)
        model_menu.bind("<<ComboboxSelected>>", self.change_model)

        # Save Directory Selection
        save_dir_button = tk.Button(
            settings_frame,
            text="Choose Save Directory",
            command=self.choose_save_directory,
            font=("Arial", 12),
            bg="lightgreen",
            fg="black",
        )
        save_dir_button.pack(side=tk.RIGHT, padx=5)

        # Progress Bar Frame
        progress_frame = Frame(self.root)
        progress_frame.pack(padx=20, pady=10, fill=tk.X)

        self.progress_label = Label(progress_frame, text="No processing", font=("Arial", 12))
        self.progress_label.pack(anchor='w')

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # Image Display Label
        self.image_label = Label(self.root, bg="white", relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Preview Frame for Video
        preview_frame = Frame(self.root)
        preview_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=False)

        preview_label_title = Label(preview_frame, text="Preview Processed Frames:", font=("Arial", 12))
        preview_label_title.pack(anchor='w')

        self.preview_label = Label(preview_frame, bg="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def change_model(self, event):
        selected_model = self.model_var.get()
        if selected_model == self.selected_model_name:
            # No change
            return
        try:
            # Disable upload and cancel buttons during model switching
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button):
                    widget.config(state='disabled')

            # Update selected model
            self.selected_model_name = selected_model
            self.MODEL_PATH = self.MODEL_FILES[self.selected_model_name]

            # Load the new model
            self.load_model()

            # Re-enable upload button
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                    widget.config(state='normal')

            logging.info(f"Switched to model '{self.selected_model_name}'.")
            messagebox.showinfo("Model Changed", f"Successfully switched to '{self.selected_model_name}'.")
        except Exception as e:
            logging.error(f"Error switching model to '{selected_model}': {e}")
            messagebox.showerror("Model Loading Error", f"Error loading model '{selected_model}':\n{e}")
            # Revert to previous model selection in combobox
            self.model_var.set(self.selected_model_name)
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                    widget.config(state='normal')

    def choose_save_directory(self):
        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            self.save_directory = Path(directory)
            logging.info(f"Save directory set to: {self.save_directory}")
            messagebox.showinfo("Save Directory", f"Save directory set to:\n{self.save_directory}")

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image or Video File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        if file_path:
            logging.info(f"Selected file: {file_path}")
            self.handle_file(file_path)

    def handle_file(self, file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            self.process_image(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            # Reset progress bar
            self.progress_var.set(0)
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Processing video...")
            # Disable upload button to prevent multiple uploads
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                    widget.config(state='disabled')
            # Enable cancel button
            self.cancel_button.config(state='normal')
            # Clear the cancel event in case it was previously set
            self.cancel_event.clear()
            # Run video processing in a separate thread to prevent GUI blocking
            threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
        else:
            logging.warning("Unsupported file type selected.")
            messagebox.showwarning("Unsupported File Type", "Please select an image or video file.")

    def process_video(self, file_path):
        start_time = time.time()  # Start timing video processing
        output_filename = f"processed_{Path(file_path).stem}.mp4"
        output_path = self.save_directory / output_filename
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video {file_path}")
            self.progress_queue.put(('error', f"Cannot open video {file_path}"))
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logging.info(f"Started processing video: {file_path}")
        self.progress_queue.put(('start', total_frames))

        # Initialize detection data
        detection_counts = {}

        try:
            frame_count = 0
            # Get selected resolution
            resolution = self.resolution_var.get().split('x')
            target_width, target_height = int(resolution[0]), int(resolution[1])

            with torch.no_grad():  # Disable gradient calculation
                while cap.isOpened():
                    if self.cancel_event.is_set():
                        logging.info("Processing canceled by user.")
                        self.progress_queue.put(('canceled', "Processing canceled by user."))
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        # Resize frame for faster processing
                        resized_frame = cv2.resize(frame, (target_width, target_height))

                        # Inference on the resized frame
                        results = self.model(resized_frame)

                        # Update detection counts
                        for *box, conf, cls in results.xyxy[0]:
                            label = self.model.names[int(cls)]
                            detection_counts[label] = detection_counts.get(label, 0) + 1

                        # Render detections on the resized frame
                        results.render()

                        # Get processed frame
                        processed_frame = results.ims[0]

                        # Resize back to original size
                        processed_frame = cv2.resize(processed_frame, (width, height))

                        # Write the processed frame to the output video
                        out.write(processed_frame)

                        # Update preview
                        self.progress_queue.put(('preview', processed_frame.copy()))

                        frame_count += 1
                        # Calculate progress percentage
                        progress = int((frame_count / total_frames) * 100)
                        self.progress_queue.put(('progress', progress))
                    except Exception as e:
                        logging.error(f"Error during video frame processing: {e}")
                        self.progress_queue.put(('error', f"Frame processing error: {e}"))
                        break
        except Exception as e:
            logging.error(f"Error during video processing: {e}")
            self.progress_queue.put(('error', f"Video processing error: {e}"))
        finally:
            cap.release()
            out.release()
            end_time = time.time()  # End timing
            processing_time = end_time - start_time

            if not self.cancel_event.is_set():
                logging.info(f"Processed video saved to: {output_path}")
                logging.info(f"Video processing time: {processing_time:.2f} seconds")
                self.progress_queue.put(('complete', str(output_path)))
                self.export_detection_data(output_path, detection_counts)
                # Try opening the processed video
                try:
                    if platform.system() == "Windows":
                        os.startfile(str(output_path))
                    elif platform.system() == "Darwin":  # macOS
                        os.system(f"open '{output_path}'")
                    else:  # Linux and others
                        os.system(f"xdg-open '{output_path}'")
                except Exception as e:
                    logging.error(f"Error opening the processed video: {e}")
                    self.progress_queue.put(('error', f"Error opening video: {e}"))
                # Also update the progress label to show processing time
                self.progress_queue.put(('info', f"Processing complete! Time: {processing_time:.2f} s"))
            else:
                # If processing was canceled, delete the partially processed video
                if output_path.exists():
                    try:
                        os.remove(str(output_path))
                        logging.info(f"Partially processed video deleted: {output_path}")
                    except Exception as e:
                        logging.error(f"Error deleting partially processed video: {e}")
                        self.progress_queue.put(('error', f"Error deleting video: {e}"))

    def process_image(self, file_path):
        start_time = time.time()  # Start timing image processing
        image = cv2.imread(str(file_path))
        if image is None:
            logging.error(f"Unable to load image from {file_path}")
            self.progress_queue.put(('error', f"Unable to load image from {file_path}"))
            self.progress_label.config(text="Error: Unable to load image.")
            return
        try:
            with torch.no_grad():  # Disable gradient calculation
                # Inference on the image
                results = self.model(image)
                results.render()
                processed_img_path = self.save_directory / Path(file_path).name
                cv2.imwrite(str(processed_img_path), results.ims[0])
                logging.info(f"Processed image saved to: {processed_img_path}")

                end_time = time.time()  # End timing
                processing_time = end_time - start_time
                logging.info(f"Image processing time: {processing_time:.2f} seconds")

                self.show_image_on_gui(processed_img_path)
                self.progress_label.config(text=f"Image processing complete! Time: {processing_time:.2f} s")
                self.progress_var.set(100)
                self.progress_bar['value'] = 100
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            self.progress_queue.put(('error', f"Inference error: {e}"))
            self.progress_label.config(text=f"Error: {e}")

    def show_image_on_gui(self, image_path):
        try:
            resized_image = self.resize_image(image_path)
            self.image_label.config(image=resized_image)
            self.image_label.image = resized_image
        except Exception as e:
            logging.error(f"Error displaying image: {e}")
            self.progress_queue.put(('error', f"Display error: {e}"))

    def resize_image(self, image_path, max_width=800, max_height=600):
        img = Image.open(image_path)
        original_width, original_height = img.size
        scaling_factor = min(max_width / original_width, max_height / original_height, 1)
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        resample_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
        img = img.resize((new_width, new_height), resample_method)
        return ImageTk.PhotoImage(img)

    def handle_drop(self, event):
        # Handle multiple files dropped at once
        files = self.root.splitlist(event.data)
        for file_path in files:
            self.handle_file(file_path)

    def cancel_processing(self):
        self.cancel_event.set()
        self.progress_label.config(text="Cancelling...")
        self.cancel_button.config(state='disabled')
        logging.info("Cancellation requested by user.")

    def export_detection_data(self, video_path, detection_counts):
        csv_filename = self.save_directory / f"detection_data_{Path(video_path).stem}.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['Object', 'Count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                for obj, count in detection_counts.items():
                    writer.writerow({'Object': obj, 'Count': count})
            logging.info(f"Detection data exported to: {csv_filename}")
        except Exception as e:
            logging.error(f"Error exporting detection data: {e}")
            self.progress_queue.put(('error', f"Error exporting detection data: {e}"))

    def update_progress(self):
        try:
            while not self.progress_queue.empty():
                msg_type, data = self.progress_queue.get_nowait()
                if msg_type == 'start':
                    total_frames = data
                    self.progress_bar.config(mode='determinate')
                    self.progress_bar['maximum'] = 100
                    self.progress_label.config(text="Processing video...")
                elif msg_type == 'progress':
                    progress = data
                    self.progress_var.set(progress)
                    self.progress_bar['value'] = progress
                    self.progress_label.config(text=f"Processing video... {progress}%")
                elif msg_type == 'complete':
                    processed_path = data
                    self.progress_var.set(100)
                    self.progress_bar['value'] = 100
                    self.progress_label.config(text="Processing complete!")
                    logging.info(f"Processed video saved to: {processed_path}")
                    # Re-enable upload button
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                            widget.config(state='normal')
                    # Disable cancel button
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'canceled':
                    self.progress_label.config(text=data)
                    logging.info(data)
                    # Re-enable upload button
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                            widget.config(state='normal')
                    # Disable cancel button
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'error':
                    error_message = data
                    self.progress_label.config(text=f"Error: {error_message}")
                    logging.error(f"Error: {error_message}")
                    # Re-enable upload button
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Upload File":
                            widget.config(state='normal')
                    # Disable cancel button
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'preview':
                    frame = data
                    self.update_preview(frame)
                elif msg_type == 'info':
                    # Additional info messages, like processing time
                    self.progress_label.config(text=data)
        except queue.Empty:
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.update_progress)

    def update_preview(self, frame):
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((500, 400), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.preview_label.config(image=imgtk)
            self.preview_label.image = imgtk
        except Exception as e:
            logging.error(f"Error updating preview: {e}")

    def run(self):
        logging.info("GUI loop started.")
        self.root.mainloop()
        # Restore the original PosixPath
        pathlib.PosixPath = self.temp
        logging.info("Application closed.")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AerialObjectDetector(root)
    app.run()
