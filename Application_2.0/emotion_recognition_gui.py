import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import os
import sys

# Import from the updated emotion recognition app
from emotion_recognition_app import AcademicEmotionRecognizer, process_video_file

class EmotionRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Emotion Recognition System")
        self.root.geometry("1400x900")
        
        self.recognizer = None
        self.model_loaded = False
        self.model_path = None  # Store the actual model file path
        self.is_running = False
        self.cap = None
        self.current_frame = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Academic Emotion Recognition System", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model loading
        ttk.Label(control_frame, text="1. Load Model", font=('Helvetica', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        ttk.Button(control_frame, text="Select Model File (.h5)", 
                  command=self.load_model).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.model_status = ttk.Label(control_frame, text="No model loaded", foreground="red")
        self.model_status.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # Live capture
        ttk.Label(control_frame, text="2. Live Webcam", font=('Helvetica', 12, 'bold')).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.start_btn = ttk.Button(control_frame, text="Start Webcam", 
                                    command=self.start_webcam, state='disabled')
        self.start_btn.grid(row=5, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Webcam", 
                                   command=self.stop_webcam, state='disabled')
        self.stop_btn.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.record_btn = ttk.Button(control_frame, text="Start Recording", 
                                     command=self.toggle_recording, state='disabled')
        self.record_btn.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.recording_status = ttk.Label(control_frame, text="Not recording", foreground="gray")
        self.recording_status.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # Video processing
        ttk.Label(control_frame, text="3. Process Video File", font=('Helvetica', 12, 'bold')).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        ttk.Button(control_frame, text="Select Video to Process", 
                  command=self.process_video).grid(row=10, column=0, columnspan=2, 
                                                   sticky=(tk.W, tk.E), pady=5)
        
        self.process_status = ttk.Label(control_frame, text="", foreground="blue")
        self.process_status.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Emotion scores display
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        ttk.Label(control_frame, text="Current Emotions", font=('Helvetica', 12, 'bold')).grid(
            row=13, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.emotion_labels = {}
        emotions = ['Boredom', 'Confusion', 'Frustration', 'Engagement']
        for i, emotion in enumerate(emotions):
            ttk.Label(control_frame, text=f"{emotion}:").grid(
                row=14+i, column=0, sticky=tk.W, pady=2)
            label = ttk.Label(control_frame, text="--", font=('Courier', 10))
            label.grid(row=14+i, column=1, sticky=tk.E, pady=2)
            self.emotion_labels[emotion] = label
        
        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_model(self):
        """Load the emotion recognition model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                self.recognizer = AcademicEmotionRecognizer(model_path)
                self.model_loaded = True
                self.model_path = model_path  # Store the model path
                self.model_status.config(text=f"Model loaded: {os.path.basename(model_path)}", 
                                        foreground="green")
                self.start_btn.config(state='normal')
                self.status_bar.config(text=f"Model loaded successfully: {model_path}")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.status_bar.config(text=f"Error loading model: {str(e)}")
    
    def start_webcam(self):
        """Start the webcam feed"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        self.cap = cv2.VideoCapture(self.recognizer.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam!")
            return
        
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.record_btn.config(state='normal')
        self.status_bar.config(text="Webcam started")
        
        # Start update loop
        self.update_frame()
    
    def stop_webcam(self):
        """Stop the webcam feed"""
        self.is_running = False
        
        if self.recognizer.is_recording:
            self.toggle_recording()
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.record_btn.config(state='disabled')
        self.video_label.config(image='')
        self.status_bar.config(text="Webcam stopped")
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recognizer.is_recording:
            output_file = self.recognizer.start_recording()
            self.recording_status.config(text=f"Recording: {output_file}", foreground="red")
            self.record_btn.config(text="Stop Recording")
            self.status_bar.config(text=f"Recording started: {output_file}")
        else:
            json_file = self.recognizer.stop_recording(
                self.recording_status.cget("text").replace("Recording: ", "")
            )
            self.recording_status.config(text="Not recording", foreground="gray")
            self.record_btn.config(text="Start Recording")
            self.status_bar.config(text=f"Recording stopped. Annotations saved: {json_file}")
            messagebox.showinfo("Recording Saved", 
                              f"Recording and annotations saved successfully!\n\n"
                              f"Check the current directory for the files.")
    
    def update_frame(self):
        """Update video frame"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        display_frame = frame.copy()
        
        # Detect face and process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.recognizer.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )
        
        predictions = None
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            mx = int(w * self.recognizer.MARGIN_RATIO)
            my = int(h * self.recognizer.MARGIN_RATIO)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(frame.shape[1], x + w + mx)
            y2 = min(frame.shape[0], y + h + my)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, 
                                      (self.recognizer.FRAME_SIZE, self.recognizer.FRAME_SIZE))
                face_crop = face_crop.astype('float32') / 255.0
                self.recognizer.frame_buffer.append(face_crop)
                
                # Throttle predictions
                self.recognizer.frames_since_prediction += 1
                should_predict = (self.recognizer.frames_since_prediction >= self.recognizer.prediction_interval)
                
                if len(self.recognizer.frame_buffer) == self.recognizer.BUFFER_SIZE and should_predict:
                    predictions = self.recognizer.predict_emotion(self.recognizer.frame_buffer)
                    self.recognizer.last_prediction = predictions
                    self.recognizer.frames_since_prediction = 0
                    
                    # Update emotion labels
                    if predictions is not None:
                        for i, emotion in enumerate(self.recognizer.emotions):
                            score = predictions[i]
                            self.emotion_labels[emotion].config(
                                text=f"{score:.3f}",
                                foreground="green" if score > 0.5 else "black"
                            )
                    
                    # Track for recording
                    if self.recognizer.is_recording and predictions is not None:
                        emotion, confidence = self.recognizer.get_dominant_emotion(predictions)
                        
                        if self.recognizer.detect_emotion_change(emotion, confidence):
                            timestamp = self.recognizer.format_timestamp(self.recognizer.frame_count)
                            self.recognizer.emotion_timeline.append({
                                'timestamp': timestamp,
                                'frame': self.recognizer.frame_count,
                                'emotion': emotion,
                                'confidence': float(confidence),
                                'all_scores': {
                                    self.recognizer.emotions[i]: float(predictions[i]) 
                                    for i in range(len(self.recognizer.emotions))
                                }
                            })
                else:
                    # Use cached prediction for display
                    predictions = self.recognizer.last_prediction
        
        # Add buffer indicator
        cv2.putText(display_frame, 
                   f"Buffer: {len(self.recognizer.frame_buffer)}/{self.recognizer.BUFFER_SIZE}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add recording indicator
        if self.recognizer.is_recording:
            cv2.circle(display_frame, (30, 70), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (50, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if self.recognizer.video_writer:
                self.recognizer.video_writer.write(display_frame)
                self.recognizer.frame_count += 1
        
        # Convert to PhotoImage and display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (960, 540))
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def process_video(self):
        """Process a video file"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if video_path:
            self.process_status.config(text="Processing... Please wait")
            self.status_bar.config(text=f"Processing video: {video_path}")
            
            # Process in separate thread to avoid freezing GUI
            def process_thread():
                try:
                    # Use the stored model path
                    process_video_file(video_path, model_path=self.model_path)
                    self.root.after(0, lambda: self.process_status.config(
                        text="✓ Processing complete! Check 'annotated_videos' folder"))
                    self.root.after(0, lambda: self.status_bar.config(
                        text="Video processing complete"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", 
                        "Video processed successfully!\n\nCheck the 'annotated_videos' folder."))
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    full_trace = traceback.format_exc()
                    
                    # Print full error to console for debugging
                    print("\n" + "="*60)
                    print("ERROR PROCESSING VIDEO:")
                    print("="*60)
                    print(full_trace)
                    print("="*60 + "\n")
                    
                    self.root.after(0, lambda msg=error_msg: self.process_status.config(
                        text=f"✗ Error: {msg}"))
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror(
                        "Error", f"Failed to process video:\n\n{msg}\n\nCheck console for details."))
            
            thread = threading.Thread(target=process_thread)
            thread.daemon = True
            thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    root.mainloop()
