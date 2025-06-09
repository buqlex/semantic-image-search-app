import requests
import ultralytics
import os
import sys
import pathlib
import torch
import cv2
from pathlib import Path
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter as tk
from tkinter import filedialog, Label, Frame, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import platform
import warnings
import threading
import queue
import multiprocessing
import logging
import csv
import time

class BirdRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Опознавание птиц")
        self.root.geometry("1200x900")

        # Initialize logging
        logging.basicConfig(
            filename='app.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logging.info("Application started.")

        # Redirect PosixPath for Windows compatibility
        self.temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        # Model files for bird recognition
        self.MODEL_FILES = {
            "Малая модель": "yolov5s.pt",
            "Средняя модель": "yolov5m.pt",
            "Большая модель": "yolov5l.pt"
        }

        # Set default model
        self.selected_model_name = "Средняя модель"
        self.MODEL_PATH = self.MODEL_FILES[self.selected_model_name]

        # Default save directory
        self.save_directory = Path("result")
        self.save_directory.mkdir(exist_ok=True)

        warnings.simplefilter("ignore", FutureWarning)

        num_threads = multiprocessing.cpu_count()
        torch.set_num_threads(num_threads)
        logging.info(f"Set PyTorch to use {num_threads} threads.")

        self.progress_queue = queue.Queue()
        self.cancel_event = threading.Event()

        self.load_model()
        self.create_menu()
        self.create_widgets()

        self.root.after(100, self.update_progress)

    def load_model(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = Path(sys._MEIPASS)
            else:
                base_path = Path(__file__).parent

            yolov5_path = base_path / 'yolov5'

            self.model = torch.hub.load(str(yolov5_path), "custom", path=self.MODEL_PATH, source="local", force_reload=True)
            self.model.to('cpu')
            self.model.eval()
            logging.info(f"Модель '{self.selected_model_name}' успешно загружена из {self.MODEL_PATH}.")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели '{self.selected_model_name}': {e}")
            messagebox.showerror("Ошибка загрузки модели", f"Ошибка загрузки модели '{self.selected_model_name}':\n{e}")
            pathlib.PosixPath = self.temp
            self.root.destroy()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="О программе", command=self.show_about)
        helpmenu.add_separator()
        helpmenu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Меню", menu=helpmenu)
        self.root.config(menu=menubar)

    def show_about(self):
        messagebox.showinfo("О программе", "Опознавание птиц\nВерсия 1.0\n© 2024 Коломийцев Д.В.")

    def create_widgets(self):
        drop_frame = Frame(self.root, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=2)
        drop_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=False)

        drop_label = Label(
            drop_frame,
            text="Перетащите сюда изображение или видео или нажмите 'Загрузить файл' ниже",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#333",
            wraplength=1100,
            justify="center",
        )
        drop_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind("<<Drop>>", self.handle_drop)

        upload_button = tk.Button(
            self.root,
            text="Загрузить файл",
            command=self.upload_file,
            font=("Arial", 14),
            bg="lightblue",
            fg="black",
        )
        upload_button.pack(pady=10)

        self.cancel_button = tk.Button(
            self.root,
            text="Отмена",
            command=self.cancel_processing,
            font=("Arial", 14),
            bg="red",
            fg="white",
            state='disabled'
        )
        self.cancel_button.pack(pady=5)

        settings_frame = Frame(self.root)
        settings_frame.pack(padx=20, pady=10, fill=tk.X)

        resolution_label = Label(settings_frame, text="Разрешение кадра:", font=("Arial", 12))
        resolution_label.pack(side=tk.LEFT, padx=5)

        self.resolution_var = tk.StringVar(value="640x480")
        resolution_options = ["320x240", "640x480", "800x600", "1280x720"]
        resolution_menu = ttk.Combobox(settings_frame, textvariable=self.resolution_var, values=resolution_options, state="readonly")
        resolution_menu.pack(side=tk.LEFT, padx=5)

        model_label = Label(settings_frame, text="Выберите модель:", font=("Arial", 12))
        model_label.pack(side=tk.LEFT, padx=20)

        self.model_var = tk.StringVar(value=self.selected_model_name)
        model_options = list(self.MODEL_FILES.keys())
        model_menu = ttk.Combobox(settings_frame, textvariable=self.model_var, values=model_options, state="readonly")
        model_menu.pack(side=tk.LEFT, padx=5)
        model_menu.bind("<<ComboboxSelected>>", self.change_model)

        save_dir_button = tk.Button(
            settings_frame,
            text="Выбрать директорию сохранения",
            command=self.choose_save_directory,
            font=("Arial", 12),
            bg="lightgreen",
            fg="black",
        )
        save_dir_button.pack(side=tk.RIGHT, padx=5)

        progress_frame = Frame(self.root)
        progress_frame.pack(padx=20, pady=10, fill=tk.X)

        self.progress_label = Label(progress_frame, text="Нет обработки", font=("Arial", 12))
        self.progress_label.pack(anchor='w')

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        self.image_label = Label(self.root, bg="white", relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        preview_frame = Frame(self.root)
        preview_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=False)

        preview_label_title = Label(preview_frame, text="Предпросмотр обработанных кадров:", font=("Arial", 12))
        preview_label_title.pack(anchor='w')

        self.preview_label = Label(preview_frame, bg="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.bird_info_label = Label(self.root, text="", font=("Arial", 12), justify="left")
        self.bird_info_label.pack(padx=20, pady=10)

    def change_model(self, event):
        selected_model = self.model_var.get()
        if selected_model == self.selected_model_name:
            return
        try:
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button):
                    widget.config(state='disabled')

            self.selected_model_name = selected_model
            self.MODEL_PATH = self.MODEL_FILES[self.selected_model_name]
            self.load_model()

            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                    widget.config(state='normal')

            logging.info(f"Переключено на модель '{self.selected_model_name}'.")
            messagebox.showinfo("Смена модели", f"Успешно переключено на '{self.selected_model_name}'.")
        except Exception as e:
            logging.error(f"Ошибка при смене модели на '{selected_model}': {e}")
            messagebox.showerror("Ошибка загрузки модели", f"Ошибка загрузки модели '{selected_model}':\n{e}")
            self.model_var.set(self.selected_model_name)
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                    widget.config(state='normal')

    def choose_save_directory(self):
        directory = filedialog.askdirectory(title="Выберите директорию сохранения")
        if directory:
            self.save_directory = Path(directory)
            logging.info(f"Директория сохранения установлена: {self.save_directory}")
            messagebox.showinfo("Директория сохранения", f"Директория сохранения установлена:\n{self.save_directory}")

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение или видео",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Видео", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        if file_path:
            logging.info(f"Выбран файл: {file_path}")
            self.handle_file(file_path)

    def handle_file(self, file_path):
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            self.process_image(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            self.progress_var.set(0)
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Обработка видео...")
            for widget in self.root.pack_slaves():
                if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                    widget.config(state='disabled')
            self.cancel_button.config(state='normal')
            self.cancel_event.clear()
            threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
        else:
            logging.warning("Неподдерживаемый тип файла.")
            messagebox.showwarning("Неподдерживаемый тип файла", "Пожалуйста, выберите изображение или видео.")

    def process_video(self, file_path):
        start_time = time.time()
        output_filename = f"processed_{Path(file_path).stem}.mp4"
        output_path = self.save_directory / output_filename
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logging.error(f"Не удается открыть видео {file_path}")
            self.progress_queue.put(('error', f"Не удается открыть видео {file_path}"))
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logging.info(f"Начата обработка видео: {file_path}")
        self.progress_queue.put(('start', total_frames))

        detection_counts = {}

        try:
            frame_count = 0
            resolution = self.resolution_var.get().split('x')
            target_width, target_height = int(resolution[0]), int(resolution[1])

            with torch.no_grad():
                while cap.isOpened():
                    if self.cancel_event.is_set():
                        logging.info("Обработка отменена пользователем.")
                        self.progress_queue.put(('canceled', "Обработка отменена пользователем."))
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        resized_frame = cv2.resize(frame, (target_width, target_height))
                        results = self.model(resized_frame)

                        for *box, conf, cls in results.xyxy[0]:
                            label = self.model.names[int(cls)]
                            detection_counts[label] = detection_counts.get(label, 0) + 1

                        results.render()
                        processed_frame = results.ims[0]
                        processed_frame = cv2.resize(processed_frame, (width, height))
                        out.write(processed_frame)
                        self.progress_queue.put(('preview', processed_frame.copy()))

                        frame_count += 1
                        progress = int((frame_count / total_frames) * 100)
                        self.progress_queue.put(('progress', progress))
                    except Exception as e:
                        logging.error(f"Ошибка при обработке кадра видео: {e}")
                        self.progress_queue.put(('error', f"Ошибка обработки кадра: {e}"))
                        break
        except Exception as e:
            logging.error(f"Ошибка при обработке видео: {e}")
            self.progress_queue.put(('error', f"Ошибка обработки видео: {e}"))
        finally:
            cap.release()
            out.release()
            end_time = time.time()
            processing_time = end_time - start_time

            if not self.cancel_event.is_set():
                logging.info(f"Обработанное видео сохранено в: {output_path}")
                logging.info(f"Время обработки видео: {processing_time:.2f} секунд")
                self.progress_queue.put(('complete', str(output_path)))
                self.export_detection_data(output_path, detection_counts)
                try:
                    if platform.system() == "Windows":
                        os.startfile(str(output_path))
                    elif platform.system() == "Darwin":
                        os.system(f"open '{output_path}'")
                    else:
                        os.system(f"xdg-open '{output_path}'")
                except Exception as e:
                    logging.error(f"Ошибка при открытии обработанного видео: {e}")
                    self.progress_queue.put(('error', f"Ошибка при открытии видео: {e}"))
                self.progress_queue.put(('info', f"Обработка завершена! Время: {processing_time:.2f} с"))
                self.display_bird_info(detection_counts)
            else:
                if output_path.exists():
                    try:
                        os.remove(str(output_path))
                        logging.info(f"Частично обработанное видео удалено: {output_path}")
                    except Exception as e:
                        logging.error(f"Ошибка при удалении частично обработанного видео: {e}")
                        self.progress_queue.put(('error', f"Ошибка при удалении видео: {e}"))

    def process_image(self, file_path):
        start_time = time.time()
        image = cv2.imread(str(file_path))
        if image is None:
            logging.error(f"Не удается загрузить изображение из {file_path}")
            self.progress_queue.put(('error', f"Не удается загрузить изображение из {file_path}"))
            self.progress_label.config(text="Ошибка: Не удается загрузить изображение.")
            return
        try:
            with torch.no_grad():
                results = self.model(image)
                results.render()
                processed_img_path = self.save_directory / Path(file_path).name
                cv2.imwrite(str(processed_img_path), results.ims[0])
                logging.info(f"Обработанное изображение сохранено в: {processed_img_path}")

                end_time = time.time()
                processing_time = end_time - start_time
                logging.info(f"Время обработки изображения: {processing_time:.2f} секунд")

                self.show_image_on_gui(processed_img_path)
                self.progress_label.config(text=f"Обработка изображения завершена! Время: {processing_time:.2f} с")
                self.progress_var.set(100)
                self.progress_bar['value'] = 100

                detection_counts = {}
                for *box, conf, cls in results.xyxy[0]:
                    label = self.model.names[int(cls)]
                    detection_counts[label] = detection_counts.get(label, 0) + 1
                self.display_bird_info(detection_counts)
        except Exception as e:
            logging.error(f"Ошибка во время инференса: {e}")
            self.progress_queue.put(('error', f"Ошибка инференса: {e}"))
            self.progress_label.config(text=f"Ошибка: {e}")

    def display_bird_info(self, detection_counts):
        en_to_ru = {
            "Cardinal": "Красный кардинал",
            "Sparrow": "Воробей",
            # TODO Add more bird name translations as needed
            # Вынести в отдельный файл
        }
        bird_info = "Распознанные птицы:\n"
        for bird, count in detection_counts.items():
            ru_bird = en_to_ru.get(bird, bird)
            bird_info += f"{ru_bird}: {count}\n"
        self.bird_info_label.config(text=bird_info)

    def show_image_on_gui(self, image_path):
        try:
            resized_image = self.resize_image(image_path)
            self.image_label.config(image=resized_image)
            self.image_label.image = resized_image
        except Exception as e:
            logging.error(f"Ошибка при отображении изображения: {e}")
            self.progress_queue.put(('error', f"Ошибка отображения: {e}"))

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
        files = self.root.splitlist(event.data)
        for file_path in files:
            self.handle_file(file_path)

    def cancel_processing(self):
        self.cancel_event.set()
        self.progress_label.config(text="Отмена...")
        self.cancel_button.config(state='disabled')
        logging.info("Отмена запрошена пользователем.")

    def export_detection_data(self, video_path, detection_counts):
        csv_filename = self.save_directory / f"detection_data_{Path(video_path).stem}.csv"
        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['Птица', 'Количество']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                for bird, count in detection_counts.items():
                    writer.writerow({'Птица': bird, 'Количество': count})
            logging.info(f"Данные детекции экспортированы в: {csv_filename}")
        except Exception as e:
            logging.error(f"Ошибка при экспорте данных детекции: {e}")
            self.progress_queue.put(('error', f"Ошибка при экспорте данных детекции: {e}"))

    def update_progress(self):
        try:
            while not self.progress_queue.empty():
                msg_type, data = self.progress_queue.get_nowait()
                if msg_type == 'start':
                    total_frames = data
                    self.progress_bar.config(mode='determinate')
                    self.progress_bar['maximum'] = 100
                    self.progress_label.config(text="Обработка видео...")
                elif msg_type == 'progress':
                    progress = data
                    self.progress_var.set(progress)
                    self.progress_bar['value'] = progress
                    self.progress_label.config(text=f"Обработка видео... {progress}%")
                elif msg_type == 'complete':
                    processed_path = data
                    self.progress_var.set(100)
                    self.progress_bar['value'] = 100
                    self.progress_label.config(text="Обработка завершена!")
                    logging.info(f"Обработанное видео сохранено в: {processed_path}")
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                            widget.config(state='normal')
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'canceled':
                    self.progress_label.config(text=data)
                    logging.info(data)
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                            widget.config(state='normal')
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'error':
                    error_message = data
                    self.progress_label.config(text=f"Ошибка: {error_message}")
                    logging.error(f"Ошибка: {error_message}")
                    for widget in self.root.pack_slaves():
                        if isinstance(widget, tk.Button) and widget['text'] == "Загрузить файл":
                            widget.config(state='normal')
                    self.cancel_button.config(state='disabled')
                elif msg_type == 'preview':
                    frame = data
                    self.update_preview(frame)
                elif msg_type == 'info':
                    self.progress_label.config(text=data)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_progress)

    def update_preview(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((500, 400), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.preview_label.config(image=imgtk)
            self.preview_label.image = imgtk
        except Exception as e:
            logging.error(f"Ошибка при обновлении предпросмотра: {e}")

    def run(self):
        logging.info("GUI loop started.")
        self.root.mainloop()
        pathlib.PosixPath = self.temp
        logging.info("Application closed.")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = BirdRecognizer(root)
    app.run()
