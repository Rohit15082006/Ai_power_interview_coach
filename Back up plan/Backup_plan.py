import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk 
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
import sounddevice as sd
import wavio
import re
import cv2
import threading
import queue
import time
import os
import json
import hashlib
import time

class InterviewCoach:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stopwords = set(stopwords.words('english'))

        self.professional_words = {
            'accomplished', 'achieved', 'analyzed', 'coordinated', 'created',
            'delivered', 'developed', 'enhanced', 'executed', 'improved',
            'initiated', 'launched', 'managed', 'optimized', 'organized',
            'planned', 'resolved', 'spearheaded', 'streamlined', 'success'
        }

        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
            'sort of', 'kind of', 'so', 'well', 'just', 'stuff', 'things'
        }

        self.sample_rate = 44100  # Hz
        self.duration = 60  # seconds
        self.recording_in_progress = False
        
        # New variables for pause/resume functionality
        self.recording_paused = False
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        self.remaining_time = self.duration
        self.recorded_frames = []
        self.recording_filename = "interview_response.wav"

    def record_voice(self, filename="interview_response.wav", callback=None):
        """Start or resume recording audio in a separate thread"""
        self.recording_in_progress = True
        self.recording_paused = False
        self.recording_filename = filename
    
        # Clear stop event
        self.stop_recording_event.clear()
    
        # Define record_thread function before referring to it
        # Replace the record_thread function with this fixed version
        def record_thread():
            print(f"Recording your answer for {self.remaining_time} seconds...")
            print("Speak now!")
            
            # Calculate number of frames needed
            frames_to_record = int(self.remaining_time * self.sample_rate)
            
            # Create empty array to store frames or use existing frames
            if len(self.recorded_frames) == 0:  # Check if recorded_frames is empty using len()
                self.recorded_frames = np.zeros((frames_to_record, 1), dtype=np.float32)
                recorded_frames_count = 0
            else:
                # Calculate how many frames we've already recorded
                recorded_frames_count = len(self.recorded_frames)
                # Create a new array with room for the remaining frames
                total_frames = recorded_frames_count + frames_to_record
                new_frames = np.zeros((total_frames, 1), dtype=np.float32)
                # Copy existing frames
                new_frames[:recorded_frames_count] = self.recorded_frames
                self.recorded_frames = new_frames
            
            stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=None)
            stream.start()
            
            start_time = time.time()
            elapsed_time = 0
            
            try:
                while elapsed_time < self.remaining_time and not self.stop_recording_event.is_set():
                    # Read from stream
                    chunk, overflowed = stream.read(int(0.1 * self.sample_rate))  # Read 100ms chunks
                    
                    # Calculate how many frames we can store
                    frames_available = min(len(chunk), frames_to_record - int(elapsed_time * self.sample_rate))
                    
                    if frames_available > 0:
                        # Store the frames
                        frame_idx = recorded_frames_count + int(elapsed_time * self.sample_rate)
                        
                        # Ensure the indices are valid
                        if frame_idx >= 0 and frame_idx + frames_available <= len(self.recorded_frames):
                            end_idx = frame_idx + frames_available
                            self.recorded_frames[frame_idx:end_idx] = chunk[:frames_available]
                    
                    elapsed_time = time.time() - start_time
            finally:
                stream.stop()
                stream.close()
            
            # Handle pause or complete
            if self.stop_recording_event.is_set() and self.recording_paused:
                # Update remaining time if paused
                self.remaining_time -= elapsed_time
                print(f"Recording paused. {self.remaining_time:.1f} seconds remaining.")
                self.recording_in_progress = False
            else:
                # If we finished or cancelled (not paused), save and process
                if not self.recording_paused:  # If not paused, then we're done or cancelled
                    if not self.stop_recording_event.is_set():  # Normal completion
                        # Calculate actual recorded duration
                        actual_recorded_time = elapsed_time + (recorded_frames_count / self.sample_rate)
                        # Trim to actual length to avoid trailing zeros
                        actual_frames = int(actual_recorded_time * self.sample_rate)
                        actual_frames = min(actual_frames, len(self.recorded_frames))  # Ensure we don't go out of bounds
                        
                        wavio.write(filename, self.recorded_frames[:actual_frames], self.sample_rate, sampwidth=2)
                        print(f"Recording saved to {filename}")
                        
                        # Reset for next recording
                        self.remaining_time = self.duration
                        self.recorded_frames = []
                        self.recording_in_progress = False
                        
                        if callback:
                            callback(filename)
                    else:  # Cancelled
                        # Reset without saving
                        self.remaining_time = self.duration
                        self.recorded_frames = []
                        self.recording_in_progress = False
                        
                        # Delete file if it exists
                        if os.path.exists(filename):
                            try:
                                os.remove(filename)
                                print(f"Recording cancelled and {filename} deleted.")
                            except:
                                print(f"Failed to delete {filename}.")
                        
                        if callback:
                            callback(None)
    
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=record_thread)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        return filename
    
    def pause_recording(self):
        """Pause the current recording"""
        if self.recording_in_progress and not self.recording_paused:
            self.recording_paused = True
            self.stop_recording_event.set()
            print("Pausing recording...")
            return True
        return False
    
    def resume_recording(self, callback=None):
        """Resume a paused recording"""
        if self.recording_paused:
            # Resume recording where we left off
            return self.record_voice(self.recording_filename, callback)
        return False
    
    def cancel_recording(self):
        """Cancel the recording and discard data"""
        if self.recording_in_progress:
            self.stop_recording_event.set()
            self.recording_paused = False  # Not paused, just cancelled
            print("Cancelling recording...")
            return True
        return False
            
    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        with sr.AudioFile(audio_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Transcription complete!")
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
                return ""
    

    def analyze_tone(self, text):
        if not text:
            return {
                'score': 0,
                'sentiment': 'neutral',
                'feedback': 'No speech detected to analyze tone.'
            }

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        feedback = ""
        if sentiment == 'positive':
            feedback = "Your tone is positive and enthusiastic, which is great for an interview. Keep up the energy!"
            if compound_score > 0.5:
                feedback += " However, be careful not to come across as overly enthusiastic as it might seem insincere."
        elif sentiment == 'negative':
            feedback = "Your tone comes across as somewhat negative. Try to use more positive language and emphasize your strengths and achievements."
        else:
            feedback = "Your tone is neutral. While this is professional, try to inject some enthusiasm when discussing your achievements or interest in the role."

        return {
            'score': compound_score,
            'sentiment': sentiment,
            'feedback': feedback
        }

    def analyze_word_choice(self, text):
        if not text:
            return {
                'professional_word_count': 0,
                'filler_word_count': 0,
                'professional_words_used': [],
                'filler_words_used': [],
                'feedback': 'No speech detected to analyze word choice.'
            }

        words = nltk.word_tokenize(text.lower())
        professional_words_used = [word for word in words if word in self.professional_words]
        filler_words_used = [filler for filler in self.filler_words if filler in text.lower()]

        feedback = ""
        if professional_words_used:
            feedback += f"Good use of professional language! Words like {', '.join(professional_words_used[:3])} strengthen your responses. "
        else:
            feedback += "Consider incorporating more professional language to highlight your skills and achievements. "

        if filler_words_used:
            feedback += f"Try to reduce filler words/phrases like {', '.join(filler_words_used[:3])}. These can make you sound less confident."
        else:
            feedback += "You've done well avoiding filler words, which makes your speech sound more confident and prepared."

        return {
            'professional_word_count': len(professional_words_used),
            'filler_word_count': len(filler_words_used),
            'professional_words_used': professional_words_used,
            'filler_words_used': filler_words_used,
            'feedback': feedback
        }

    def analyze_confidence(self, text, tone_analysis):
        if not text:
            return {
                'confidence_score': 0,
                'feedback': 'No speech detected to analyze confidence.'
            }

        confidence_score = 5  # Base score out of 10
        sentiment_score = tone_analysis['score']
        if sentiment_score > 0:
            confidence_score += sentiment_score * 2
        elif sentiment_score < -0.2:
            confidence_score -= abs(sentiment_score) * 2

        hesitation_patterns = [
            r'\bI think\b', r'\bmaybe\b', r'\bpossibly\b', r'\bperhaps\b',
            r'\bI guess\b', r'\bsort of\b', r'\bkind of\b', r'\bI hope\b',
            r'\bI\'m not sure\b', r'\bI don\'t know\b'
        ]

        hesitation_count = sum(len(re.findall(pattern, text.lower())) for pattern in hesitation_patterns)
        confidence_score -= hesitation_count * 0.5

        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0

        if avg_sentence_length > 20:
            confidence_score += 1
        elif avg_sentence_length < 8:
            confidence_score -= 1

        confidence_score = max(0, min(10, confidence_score))

        if confidence_score >= 8:
            feedback = "You sound very confident. Your delivery is strong and assertive."
        elif confidence_score >= 6:
            feedback = "You sound reasonably confident. With a few adjustments, you could project even more authority."
        elif confidence_score >= 4:
            feedback = "Your confidence level seems moderate. Try speaking more assertively and avoiding hesitant language."
        else:
            feedback = "You may want to work on projecting more confidence. Try reducing hesitant phrases and speaking with more conviction."

        return {
            'confidence_score': confidence_score,
            'feedback': feedback
        }

    def provide_comprehensive_feedback(self, analysis_results):
        tone = analysis_results['tone']
        word_choice = analysis_results['word_choice']
        confidence = analysis_results['confidence']

        feedback_text = "\n" + "=" * 50 + "\n"
        feedback_text += "INTERVIEW RESPONSE EVALUATION\n"
        feedback_text += "=" * 50 + "\n\n"

        feedback_text += "TONE ANALYSIS:\n"
        feedback_text += f"Sentiment: {tone['sentiment']} (Score: {tone['score']:.2f})\n"
        feedback_text += f"Feedback: {tone['feedback']}\n\n"

        feedback_text += "WORD CHOICE ANALYSIS:\n"
        feedback_text += f"Professional words used: {word_choice['professional_word_count']}\n"
        if word_choice['professional_words_used']:
            feedback_text += f"Examples: {', '.join(list(set(word_choice['professional_words_used']))[:5])}\n"

        feedback_text += f"Filler words/phrases used: {word_choice['filler_word_count']}\n"
        if word_choice['filler_words_used']:
            feedback_text += f"Examples: {', '.join(list(set(word_choice['filler_words_used']))[:5])}\n"

        feedback_text += f"Feedback: {word_choice['feedback']}\n\n"

        feedback_text += "CONFIDENCE ASSESSMENT:\n"
        feedback_text += f"Confidence Score: {confidence['confidence_score']:.1f}/10\n"
        feedback_text += f"Feedback: {confidence['feedback']}\n\n"

        avg_score = (tone['score'] + 1) * 5 + confidence['confidence_score']
        avg_score /= 2

        if avg_score >= 8:
            feedback_text += "Excellent interview response! You presented yourself very well.\n"
        elif avg_score >= 6:
            feedback_text += "Good interview response. With some minor improvements, you'll make an even stronger impression.\n"
        elif avg_score >= 4:
            feedback_text += "Acceptable interview response. Focus on the improvement areas mentioned above.\n"
        else:
            feedback_text += "Your interview response needs improvement. Consider practicing more with the suggestions provided.\n"

        feedback_text += "\nAREAS TO FOCUS ON:\n"
        improvement_areas = []

        if tone['score'] < 0:
            improvement_areas.append("Using more positive language")
        if word_choice['filler_word_count'] > 3:
            improvement_areas.append("Reducing filler words/phrases")
        if word_choice['professional_word_count'] < 2:
            improvement_areas.append("Incorporating more professional vocabulary")
        if confidence['confidence_score'] < 5:
            improvement_areas.append("Building confidence in delivery")

        if improvement_areas:
            for i, area in enumerate(improvement_areas, 1):
                feedback_text += f"{i}. {area}\n"
        else:
            feedback_text += "Great job! Keep practicing to maintain your strong performance.\n"

        feedback_text += "=" * 50 + "\n"

        return feedback_text

    def analyze_text_input(self, text):
        tone_analysis = self.analyze_tone(text)
        word_choice_analysis = self.analyze_word_choice(text)
        confidence_analysis = self.analyze_confidence(text, tone_analysis)

        analysis_results = {
            'tone': tone_analysis,
            'word_choice': word_choice_analysis,
            'confidence': confidence_analysis,
            'text': text
        }

        return analysis_results

    def start_voice_analysis(self, callback=None):
        """Start voice analysis with callback for async processing"""
        def process_recording(audio_file):
            text = self.transcribe_audio(audio_file)
            if text:
                print(f"\nTranscribed Text:\n{text}\n")

                tone_analysis = self.analyze_tone(text)
                word_choice_analysis = self.analyze_word_choice(text)
                confidence_analysis = self.analyze_confidence(text, tone_analysis)

                analysis_results = {
                    'tone': tone_analysis,
                    'word_choice': word_choice_analysis,
                    'confidence': confidence_analysis,
                    'text': text
                }

                if callback:
                    callback(analysis_results)
                return analysis_results
            else:
                print("No speech detected or transcription failed.")
                if callback:
                    callback(None)
                return None

        try:
            # Start recording with a callback to process the recording when done
            self.record_voice(callback=lambda audio_file: threading.Thread(
                target=process_recording, args=(audio_file,)).start())
        except Exception as e:
            print(f"An error occurred: {e}")
            if callback:
                callback(None)
            return None

class CameraHandler:
    """Separate class to handle camera operations"""
    def __init__(self):
        self.capture = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
        self.face_positions = []
        # Load the pre-trained Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def start(self):
        if self.running:
            return
            
        self.capture = cv2.VideoCapture(0)
        self.running = True
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
    def stop(self):
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
            
    def _camera_loop(self):
        """Camera capture loop running in separate thread"""
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Process the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Track the y position of face center for posture analysis
                self.face_positions.append(y + h/2)
                # Keep only the last 10 positions for analysis
                if len(self.face_positions) > 10:
                    self.face_positions.pop(0)
            
            # Convert to RGB for tkinter display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror image
            
            # Update the queue with latest frame
            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)
            else:
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put(frame_rgb)  # Put new frame
                except queue.Empty:
                    pass
                    
            time.sleep(0.01)  # Short sleep to prevent CPU overuse
            
    def get_latest_frame(self):
        """Get the latest frame if available"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
            
    def analyze_posture(self):
        """Analyze posture based on face positions"""
        if not self.face_positions:
            return "No face detected"
        
        # Calculate variance in face position
        if len(self.face_positions) > 5:
            variance = np.var(self.face_positions)
            
            if variance > 50:  # Threshold for excessive movement
                return "Poor - Too much movement"
            
            # Calculate if face position is consistently too low
            avg_position = np.mean(self.face_positions)
            
            if self.capture:
                height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                if avg_position > (height * 0.6):
                    return "Poor - Head position too low"
                elif avg_position < (height * 0.3):
                    return "Check camera position"
                else:
                    return "Good posture"
            else:
                return "Camera not initialized"
        else:
            return "Analyzing..."

class InterviewCoachGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI-Powered Interview Coach")
        
        # Set application icon (if needed)
        # master.iconbitmap('path/to/icon.ico')
        
        # Configure window
        master.geometry("800x600")
        master.minsize(800, 600)
        
        # Initialize coach and camera handler
        self.coach = InterviewCoach()
        self.camera_handler = CameraHandler()
        self.camera_handler.start()

        # Setup UI
        self.setup_ui()
        
        # Start camera update
        self.update_camera_feed()
        
        # Handle proper shutdown
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Recording status
        self.recording = False
        self.paused = False
        self.countdown_timer = None
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main layout frames
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split window into left and right panels
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left panel - Camera feed and posture analysis
        self.setup_camera_panel()
        
        # Right panel - Recording controls and results
        self.setup_control_panel()
        
    def setup_camera_panel(self):
        """Set up camera panel UI"""
        camera_frame = ttk.LabelFrame(self.left_panel, text="Posture Analysis")
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(camera_frame, width=320, height=240, bg="black")
        self.camera_canvas.pack(pady=10, padx=10)
        
        # Posture feedback label
        self.posture_label = ttk.Label(camera_frame, text="Initializing camera...")
        self.posture_label.pack(pady=5)
        
        # Posture tips
        tips_frame = ttk.LabelFrame(camera_frame, text="Posture Tips")
        tips_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tips_text = (
            "• Sit up straight with shoulders back\n"
            "• Keep your face centered in the camera\n" 
            "• Maintain eye contact with the camera\n"
            "• Avoid excessive movement\n"
            "• Use natural hand gestures when appropriate"
        )
        
        tips_label = ttk.Label(tips_frame, text=tips_text, justify=tk.LEFT)
        tips_label.pack(padx=10, pady=10)
        
    def setup_control_panel(self):
        """Set up control panel UI"""
        # Title
        title_label = ttk.Label(
            self.right_panel, 
            text="AI-Powered Interview Coach", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(self.right_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Record button 
        self.record_button = ttk.Button(
            control_frame, 
            text="Record Answer (60 sec)", 
            command=self.toggle_recording
        )
        self.record_button.pack(pady=5)
        
        # Recording buttons frame
        recording_buttons_frame = ttk.Frame(control_frame)
        recording_buttons_frame.pack(pady=5)
        
        # Pause/Resume button
        self.pause_resume_button = ttk.Button(
            recording_buttons_frame, 
            text="Pause Recording", 
            command=self.toggle_pause_resume,
            state="disabled"
        )
        self.pause_resume_button.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        self.cancel_button = ttk.Button(
            recording_buttons_frame, 
            text="Cancel Recording", 
            command=self.cancel_recording,
            state="disabled"
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        self.timer_label = ttk.Label(control_frame, text="")
        self.timer_label.pack(pady=5)
        
        # Question selection
        question_frame = ttk.LabelFrame(self.right_panel, text="Practice Questions")
        question_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.question_var = tk.StringVar()
        self.question_dropdown = ttk.Combobox(
            question_frame, 
            textvariable=self.question_var,
            state="readonly",
            width=50
        )
        
        self.questions = [
            "Tell me about yourself.",
            "What are your strengths?",
            "What are your weaknesses?",
            "Why do you want to work for this company?",
            "Where do you see yourself in five years?",
            "Why should we hire you?",
            "Tell me about a challenge you faced and how you overcame it.",
            "What's your greatest professional achievement?",
            "How do you handle stress and pressure?",
            "Do you have any questions for me?"
        ]
        
        self.question_dropdown['values'] = self.questions
        self.question_dropdown.current(0)
        self.question_dropdown.pack(padx=10, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.right_panel, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the text and scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, width=40, height=15)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Download button
        self.download_button = ttk.Button(
            results_frame,
            text="Download Report",
            command=self.download_report,
            state="disabled"  # Start disabled until there's content
        )
        self.download_button.pack(pady=10)
        
    def update_camera_feed(self):
        """Update camera feed in UI at regular intervals"""
        frame = self.camera_handler.get_latest_frame()
    
        if frame is not None:
            # Resize frame to fit canvas
            frame = cv2.resize(frame, (320, 240))
        
            # Convert OpenCV's BGR to RGB for tkinter using PIL
            from PIL import Image, ImageTk
        
            image = Image.fromarray(frame)
            photo_image = ImageTk.PhotoImage(image=image)
        
            # Update canvas with new image
            if not hasattr(self, 'photo'):
                self.photo = photo_image
                self.image_on_canvas = self.camera_canvas.create_image(
                    0, 0, image=self.photo, anchor=tk.NW
                )
            else:
                self.photo = photo_image
                self.camera_canvas.itemconfig(self.image_on_canvas, image=self.photo)
            
            # Update posture analysis
            posture = self.camera_handler.analyze_posture()
            self.posture_label.config(text=f"Posture Assessment: {posture}")
        
        # Schedule next update
        self.master.after(33, self.update_camera_feed)  # ~30 FPS
    
    def toggle_recording(self):
        """Handle recording button click"""
        if self.coach.recording_in_progress:
            # Do nothing if already recording
            return
            
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_button.config(text="Recording... (please wait)")
            self.record_button.state(['disabled'])
            
            # Enable pause and cancel buttons
            self.pause_resume_button.state(['!disabled'])
            self.cancel_button.state(['!disabled'])
            
            # Update UI to show we're recording
            self.timer_label.config(text="Recording in progress (60 seconds)")
            
            # Start recording in separate thread
            def on_recording_complete(results):
                self.master.after(0, lambda: self.handle_results(results))
                
            self.coach.start_voice_analysis(callback=on_recording_complete)
            
            # Start countdown
            self.start_countdown(60)
            
    def toggle_pause_resume(self):
        """Handle pause/resume button click"""
        if not self.recording:
            return
        
        if not self.paused:
            # Pause recording
            self.paused = True
            if self.coach.pause_recording():
                self.pause_resume_button.config(text="Resume Recording")
                if self.countdown_timer:
                    self.master.after_cancel(self.countdown_timer)
                    self.timer_label.config(text=f"Recording paused ({self.coach.remaining_time:.1f} seconds remaining)")
        else:
            # Resume recording
            self.paused = False
            self.pause_resume_button.config(text="Pause Recording")
            
            def on_recording_complete(results):
                self.master.after(0, lambda: self.handle_results(results))
                
            self.coach.resume_recording(callback=on_recording_complete)
            self.start_countdown(self.coach.remaining_time)
            
    def cancel_recording(self):
        """Cancel the current recording"""
        if self.recording:
            self.coach.cancel_recording()
            self.recording = False
            self.paused = False
            
            # Reset UI
            self.record_button.config(text="Record Answer (60 sec)")
            self.record_button.state(['!disabled'])
            self.pause_resume_button.config(text="Pause Recording")
            self.pause_resume_button.state(['disabled'])
            self.cancel_button.state(['disabled'])
            self.timer_label.config(text="Recording cancelled")
            
            if self.countdown_timer:
                self.master.after_cancel(self.countdown_timer)
                
    def start_countdown(self, seconds):
        """Start countdown timer for recording"""
        if seconds <= 0:
            return
            
        self.timer_label.config(text=f"Recording: {int(seconds)} seconds remaining")
        
        if seconds > 0:
            self.countdown_timer = self.master.after(1000, lambda: self.start_countdown(seconds - 1))
        else:
            # Timer finished, but actual recording might still be processing
            self.timer_label.config(text="Processing recording...")
            
    def handle_results(self, results):
        """Handle analysis results from the coach"""
        # Reset recording state
        self.recording = False
        self.paused = False
        
        # Reset UI
        self.record_button.config(text="Record Answer (60 sec)")
        self.record_button.state(['!disabled'])
        self.pause_resume_button.config(text="Pause Recording")
        self.pause_resume_button.state(['disabled'])
        self.cancel_button.state(['disabled'])
        self.timer_label.config(text="Recording complete")
        
        if self.countdown_timer:
            self.master.after_cancel(self.countdown_timer)
            
        # Display results
        if results and isinstance(results, dict):
            # Clear previous results
            self.results_text.delete('1.0', tk.END)
            
            # Add question
            self.results_text.insert(tk.END, f"Question: {self.question_var.get()}\n\n")
            
            # Add transcribed text
            self.results_text.insert(tk.END, f"Your Response:\n{results['text']}\n\n")
            
            # Add comprehensive feedback
            feedback = self.coach.provide_comprehensive_feedback(results)
            self.results_text.insert(tk.END, feedback)
            
            # Enable download button
            self.download_button.state(['!disabled'])
        else:
            self.results_text.delete('1.0', tk.END)
            if isinstance(results, str):
                self.results_text.insert(tk.END, "Analysis failed. Please try again.")
                self.download_button.state(['disabled'])
            else:
                self.results_text.insert(tk.END, "Recording failed. Please try again.")
                self.download_button.state(['disabled'])
            
    def download_report(self):
        """Save the analysis report to a file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Interview Analysis"
        )
        
        if filename:
            try:
                with open(filename, 'w') as file:
                    # Write question
                    file.write(f"Question: {self.question_var.get()}\n\n")
                    
                    # Write full text content
                    file.write(self.results_text.get('1.0', tk.END))
                    
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
                
    def on_closing(self):
        """Clean up resources when window closes"""
        if self.recording:
            self.coach.cancel_recording()
            
        # Stop camera
        self.camera_handler.stop()
        
        # Close the window
        self.master.destroy()

class LoginSystem:
    def __init__(self, master, on_successful_login):
        self.master = master
        self.on_successful_login = on_successful_login
        
        # Setup the main window
        self.master.title("Interview Coach - Login")
        self.master.geometry("500x400")
        self.master.minsize(500, 400)
        
        # Load existing user data
        self.users_file = "users.json"
        self.users = self.load_users()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.master, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create login/register frames
        self.setup_login_frame()
        self.setup_register_frame()
        
        # Show login frame by default
        self.show_login_frame()
    
    def load_users(self):
        """Load users from file or create empty dict if file doesn't exist"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)
    
    def setup_login_frame(self):
        """Create login frame with username and password fields"""
        self.login_frame = ttk.Frame(self.main_frame)
        
        # Title
                # Title
        title_label = ttk.Label(
            self.login_frame, 
            text="Interview Coach Login", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Username
        username_frame = ttk.Frame(self.login_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=12)
        username_label.pack(side=tk.LEFT)
        
        self.login_username = ttk.Entry(username_frame, width=30)
        self.login_username.pack(side=tk.LEFT, padx=5)
        
        # Password
        password_frame = ttk.Frame(self.login_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        password_label = ttk.Label(password_frame, text="Password:", width=12)
        password_label.pack(side=tk.LEFT)
        
        self.login_password = ttk.Entry(password_frame, show="•", width=30)
        self.login_password.pack(side=tk.LEFT, padx=5)
        
        # Login button
        login_button = ttk.Button(
            self.login_frame,
            text="Login",
            command=self.login
        )
        login_button.pack(pady=20)
        
        # Register link
        register_frame = ttk.Frame(self.login_frame)
        register_frame.pack(pady=10)
        
        register_text = ttk.Label(
            register_frame,
            text="Don't have an account? "
        )
        register_text.pack(side=tk.LEFT)
        
        register_link = ttk.Label(
            register_frame,
            text="Create one",
            foreground="blue",
            cursor="hand2"
        )
        register_link.pack(side=tk.LEFT)
        register_link.bind("<Button-1>", lambda e: self.show_register_frame())
    
    def setup_register_frame(self):
        """Create registration frame with validation"""
        self.register_frame = ttk.Frame(self.main_frame)
        
        # Title
        title_label = ttk.Label(
            self.register_frame, 
            text="Create New Account", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Full Name
        name_frame = ttk.Frame(self.register_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        name_label = ttk.Label(name_frame, text="Full Name:", width=15)
        name_label.pack(side=tk.LEFT)
        
        self.reg_fullname = ttk.Entry(name_frame, width=30)
        self.reg_fullname.pack(side=tk.LEFT, padx=5)
        
        # Username
        username_frame = ttk.Frame(self.register_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=15)
        username_label.pack(side=tk.LEFT)
        
        self.reg_username = ttk.Entry(username_frame, width=30)
        self.reg_username.pack(side=tk.LEFT, padx=5)
        
        username_hint = ttk.Label(
            self.register_frame,
            text="Username must be at least 8 characters (letters and numbers only)",
            font=("Arial", 8)
        )
        username_hint.pack(anchor=tk.W, padx=(110, 0))
        
        # Email
        email_frame = ttk.Frame(self.register_frame)
        email_frame.pack(fill=tk.X, pady=5)
        
        email_label = ttk.Label(email_frame, text="Email:", width=15)
        email_label.pack(side=tk.LEFT)
        
        self.reg_email = ttk.Entry(email_frame, width=30)
        self.reg_email.pack(side=tk.LEFT, padx=5)
        
        # Password
        password_frame = ttk.Frame(self.register_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        password_label = ttk.Label(password_frame, text="Password:", width=15)
        password_label.pack(side=tk.LEFT)
        
        self.reg_password = ttk.Entry(password_frame, show="•", width=30)
        self.reg_password.pack(side=tk.LEFT, padx=5)
        
        password_hint = ttk.Label(
            self.register_frame,
            text="Password must:\n"
                 "- Be at least 8 characters long\n"
                 "- Contain uppercase and lowercase letters\n"
                 "- Contain at least one digit\n"
                 "- Contain at least one special character (!@#$%^&*-_+=?)",
            font=("Arial", 8)
        )
        password_hint.pack(anchor=tk.W, padx=(110, 0))
        
        # Confirm Password
        confirm_frame = ttk.Frame(self.register_frame)
        confirm_frame.pack(fill=tk.X, pady=5)
        
        confirm_label = ttk.Label(confirm_frame, text="Confirm Password:", width=17)
        confirm_label.pack(side=tk.LEFT)
        
        self.reg_confirm = ttk.Entry(confirm_frame, show="•", width=30)
        self.reg_confirm.pack(side=tk.LEFT, padx=5)
        
        # Register button
        register_button = ttk.Button(
            self.register_frame,
            text="Create Account",
            command=self.register
        )
        register_button.pack(pady=10)
        
        # Back to login link
        login_frame = ttk.Frame(self.register_frame)
        login_frame.pack(pady=5)
        
        login_text = ttk.Label(
            login_frame,
            text="Already have an account? "
        )
        login_text.pack(side=tk.LEFT)
        
        login_link = ttk.Label(
            login_frame,
            text="Login",
            foreground="blue",
            cursor="hand2"
        )
        login_link.pack(side=tk.LEFT)
        login_link.bind("<Button-1>", lambda e: self.show_login_frame())
    
    def show_login_frame(self):
        """Show login frame and hide register frame"""
        self.register_frame.pack_forget()
        self.login_frame.pack(fill=tk.BOTH, expand=True)
        
    def show_register_frame(self):
        """Show register frame and hide login frame"""
        self.login_frame.pack_forget()
        self.register_frame.pack(fill=tk.BOTH, expand=True)
    
    def hash_password(self, password):
        """Simple password hashing using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_username(self, username):
        """Validates that username is at least 8 characters and alphanumeric"""
        if not username:
            return False, "Username cannot be empty"
        
        if len(username) < 8:
            return False, "Username must be at least 8 characters long"
        
        if not re.match(r'^[a-zA-Z0-9]+$', username):
            return False, "Username can only contain letters and numbers"
        
        return True, "Valid username"
    
    def validate_password(self, password):
        """Validates that password meets complex requirements"""
        if not password:
            return False, "Password cannot be empty"
        
        # Check length
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        # Check for uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for digit
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one digit"
        
        # Check for special character
        if not re.search(r'[!@#$%^&*\-_+=?]', password):
            return False, "Password must contain at least one special character (!@#$%^&*-_+=?)"
        
        return True, "Valid password"
    
    def validate_email(self, email):
        """Validates email format"""
        if not email:
            return False, "Email cannot be empty"
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False, "Invalid email format"
        
        return True, "Valid email"
    
    def login(self):
        """Handle login attempt"""
        username = self.login_username.get()
        password = self.login_password.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        # Check if username exists and password matches
        if username in self.users and self.users[username]["password"] == self.hash_password(password):
            messagebox.showinfo("Success", f"Welcome back, {username}!")
            self.master.destroy()
            self.on_successful_login()
        else:
            messagebox.showerror("Error", "Invalid username or password")
    
    def register(self):
        """Handle registration attempt with validation"""
        fullname = self.reg_fullname.get()
        username = self.reg_username.get()
        email = self.reg_email.get()
        password = self.reg_password.get()
        confirm = self.reg_confirm.get()
        
        # Validate fullname
        if not fullname:
            messagebox.showerror("Error", "Please enter your full name")
            return
        
        # Validate username
        valid_username, username_msg = self.validate_username(username)
        if not valid_username:
            messagebox.showerror("Error", username_msg)
            return
        
        # Check if username already exists
        if username in self.users:
            messagebox.showerror("Error", "Username already exists. Please choose another.")
            return
        
        # Validate email
        valid_email, email_msg = self.validate_email(email)
        if not valid_email:
            messagebox.showerror("Error", email_msg)
            return
        
        # Validate password
        valid_password, password_msg = self.validate_password(password)
        if not valid_password:
            messagebox.showerror("Error", password_msg)
            return
        
        # Validate password confirmation
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        # Create new user
        self.users[username] = {
            "fullname": fullname,
            "email": email,
            "password": self.hash_password(password)
        }
        
        # Save to file
        self.save_users()
        
        messagebox.showinfo("Success", "Account created successfully. You can now login.")
        self.show_login_frame()
        
        # Clear registration fields
        self.reg_fullname.delete(0, tk.END)
        self.reg_username.delete(0, tk.END)
        self.reg_email.delete(0, tk.END)
        self.reg_password.delete(0, tk.END)
        self.reg_confirm.delete(0, tk.END)
        
        # Pre-fill login username
        self.login_username.delete(0, tk.END)
        self.login_username.insert(0, username)
        self.login_password.focus()

def launch_main_application():
    """Start the main Interview Coach application"""
    root = tk.Tk()
    app = InterviewCoachGUI(root)
    root.mainloop()

def main():
    # Create a login window first
    login_root = tk.Tk()
    login_system = LoginSystem(login_root, launch_main_application)
    login_root.mainloop()

if __name__ == "__main__":
    main()
