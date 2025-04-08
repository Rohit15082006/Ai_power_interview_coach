import streamlit as st
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import numpy as np
import sounddevice as sd
import wavio
import re
import threading
import queue
import time
import os
import hashlib
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Database setup (using JSON file for simplicity)
USER_DB_FILE = "users.json"

def init_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)

def hash_password(password):
    """Create a SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password meets requirements"""
    # At least 8 characters, one uppercase, one number, one special character
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def validate_username(username):
    """Validate username meets requirements"""
    if len(username) < 8:
        return False, "Username must be at least 8 characters"
    return True, "Username is valid"

def user_exists(username):
    """Check if a user already exists"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    return username in users

def add_user(username, password):
    """Add a new user to the database"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    users[username] = hash_password(password)
    
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def authenticate_user(username, password):
    """Authenticate a user"""
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
    
    if username in users and users[username] == hash_password(password):
        return True
    return False

# Video frame transformer for webcam
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Add non-invasive interview tips at the top of the video
        h, w = img.shape[:2]
        tip_text = "Maintain eye contact and show confidence!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 30)
        cv2.putText(img, tip_text, position, font, 0.7, (0, 255, 0), 2)
        
        return img

class InterviewCoach:
    def __init__(self):
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
        self.duration = 30  # seconds
        self.recording_in_progress = False
        self.recording_thread = None
        self.stop_recording = False

    def record_voice(self, filename="interview_response.wav"):
        """Record audio"""
        self.recording_in_progress = True
        self.stop_recording = False
        
        print(f"Recording your answer for {self.duration} seconds...")
        print("Speak now!")
        
        # Calculate total samples
        total_samples = int(self.duration * self.sample_rate)
        
        # Create array to store audio data
        recording = np.zeros((total_samples, 1))
        
        # Record in chunks to allow for stopping
        chunk_size = int(0.1 * self.sample_rate)  # 0.1 second chunks
        recorded_samples = 0
        
        # Start recording
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
        stream.start()
        
        while recorded_samples < total_samples and not self.stop_recording:
            # Calculate remaining samples in this chunk
            samples_to_record = min(chunk_size, total_samples - recorded_samples)
            
            # Record chunk
            data, overflowed = stream.read(samples_to_record)
            
            # Store in our recording array
            recording[recorded_samples:recorded_samples + len(data)] = data
            
            # Update position
            recorded_samples += len(data)
            
        # Stop and close the stream
        stream.stop()
        stream.close()
        
        # If we have any recorded data and not just stopped immediately
        if recorded_samples > self.sample_rate * 0.5:  # At least half a second
            # Trim recording to actual length
            recording = recording[:recorded_samples]
            
            # Save recording
            wavio.write(filename, recording, self.sample_rate, sampwidth=2)
            print(f"Recording saved to {filename}")
            return filename
        else:
            print("Recording canceled or too short")
            return None

    def transcribe_audio(self, audio_file):
        print("Transcribing audio...")
        if audio_file is None:
            return ""
            
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
            feedback_text += f"Examples: {', '.join(word_choice['professional_words_used'][:3])}\n"

        feedback_text += f"Filler words/phrases used: {word_choice['filler_word_count']}\n"
        if word_choice['filler_words_used']:
            feedback_text += f"Examples: {', '.join(word_choice['filler_words_used'][:3])}\n"

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

def create_login_page():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        if username and password:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

def create_signup_page():
    st.subheader("Create an Account")
    
    username = st.text_input("Username (minimum 8 characters)", key="signup_username")
    password = st.text_input("Password", type="password", help="Password must contain at least 8 characters, one uppercase letter, one number, and one special character", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.button("Sign Up"):
        if not username or not password or not confirm_password:
            st.warning("Please fill in all fields")
            return
            
        valid_username, username_msg = validate_username(username)
        if not valid_username:
            st.error(username_msg)
            return
            
        valid_password, password_msg = validate_password(password)
        if not valid_password:
            st.error(password_msg)
            return
            
        if password != confirm_password:
            st.error("Passwords do not match")
            return
            
        if user_exists(username):
            st.error("Username already exists. Please choose another one.")
            return
            
        add_user(username, password)
        st.success("Account created successfully! Please log in.")
        st.session_state.show_login = True
        st.rerun()

def main_app():
    st.title("AI-Powered Interview Coach")
    st.write(f"Welcome, {st.session_state.username}! Practice your interview skills and receive feedback on your responses.")
    
    # Create a sidebar with options
    menu = ["Practice Interview", "View Tips", "Settings", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Practice Interview":
        st.header("Practice Interview")
        
        # Initialize camera_active state if it doesn't exist
        if "camera_active" not in st.session_state:
            st.session_state.camera_active = True
        
        # Webcam feedback section - only show when camera is active
        if st.session_state.camera_active:
            st.subheader("Video Feedback")
            webrtc_ctx = webrtc_streamer(
                key="interview-webcam",
                video_transformer_factory=VideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": True},
            )
            
            if webrtc_ctx.video_transformer:
                st.info("Webcam is active. You can now see yourself as you would appear in an interview.")
                st.markdown("""
                **Tips for good video presence:**
                - Maintain eye contact with the camera
                - Sit up straight with good posture
                - Ensure your face is well-lit
                - Keep a neutral background
                """)
                
                # Add manual camera control
                if st.button("Stop Camera"):
                    st.session_state.camera_active = False
                    st.rerun()
        else:
            if st.button("Start Camera"):
                st.session_state.camera_active = True
                st.rerun()
        
        # Interview practice section
        st.subheader("Voice Analysis")
        coach = InterviewCoach()
        
        # Sample interview questions
        questions = [
            "Tell me about yourself.",
            "What are your greatest strengths?",
            "What is your biggest weakness?",
            "Why do you want to work at our company?",
            "Describe a challenging situation and how you handled it.",
            "Where do you see yourself in 5 years?",
            "Custom question (type below)"
        ]
        
        selected_question = st.selectbox("Select an interview question to answer:", questions)
        
        if selected_question == "Custom question (type below)":
            custom_question = st.text_input("Enter your custom question:")
            if custom_question:
                st.write(f"Question: {custom_question}")
        else:
            st.write(f"Question: {selected_question}")
            
        # Recording duration selection
        duration = st.selectbox("Select Recording Duration (seconds)", [15, 30, 45, 60, 90, 120], index=1)
        coach.duration = duration
        
        # Record button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Record Answer"):
                with st.spinner("Recording in progress..."):
                    audio_file = coach.record_voice()
                    if audio_file:
                        st.success("Recording complete! Transcribing...")
                        text = coach.transcribe_audio(audio_file)
                        if text:
                            st.session_state.transcribed_text = text
                            st.session_state.analysis_results = coach.analyze_text_input(text)
                            # Stop camera after analysis
                            st.session_state.camera_active = False
                            st.rerun()
                        else:
                            st.error("Transcription failed. Please try again.")
                    else:
                        st.error("Recording was canceled or too short.")
        
        with col2:
            if st.button("Text Input Instead"):
                st.session_state.show_text_input = True
                st.rerun()
                
        if "show_text_input" in st.session_state and st.session_state.show_text_input:
            text_input = st.text_area("Type your answer instead:", height=150)
            if st.button("Analyze Text"):
                if text_input:
                    st.session_state.transcribed_text = text_input
                    st.session_state.analysis_results = coach.analyze_text_input(text_input)
                    st.session_state.show_text_input = False
                    # Stop camera after analysis
                    st.session_state.camera_active = False
                    st.rerun()
                else:
                    st.warning("Please enter some text to analyze.")
        
        # Display transcription and analysis if available
        if "transcribed_text" in st.session_state and "analysis_results" in st.session_state:
            st.subheader("Your Answer")
            st.write(st.session_state.transcribed_text)
            
            st.subheader("Analysis Results")
            feedback = coach.provide_comprehensive_feedback(st.session_state.analysis_results)
            st.text(feedback)
            
            if st.button("Clear Results and Restart"):
                if "transcribed_text" in st.session_state:
                    del st.session_state.transcribed_text
                if "analysis_results" in st.session_state:
                    del st.session_state.analysis_results
                # Optionally restart camera
                st.session_state.camera_active = True
                st.rerun()
    
    elif choice == "View Tips":
        st.header("Interview Tips")
        st.markdown("""
        ### General Interview Tips
        
        **Before the Interview:**
        - Research the company thoroughly
        - Practice common interview questions
        - Prepare specific examples of your achievements
        - Plan your outfit the day before
        
        **During the Interview:**
        - Arrive 10-15 minutes early
        - Make eye contact and offer a firm handshake
        - Use the STAR method for behavioral questions (Situation, Task, Action, Result)
        - Ask thoughtful questions about the company and role
        
        **Virtual Interview Tips:**
        - Test your technology beforehand
        - Choose a quiet, well-lit space with a neutral background
        - Dress professionally (even below the camera)
        - Look at the camera, not the screen, to maintain "eye contact"
        """)
    
    elif choice == "Settings":
        st.header("Settings")
        st.subheader("Account Information")
        st.write(f"Username: {st.session_state.username}")
        
        if st.button("Change Password"):
            st.session_state.change_password = True
            st.rerun()
            
        if "change_password" in st.session_state and st.session_state.change_password:
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if not current_password or not new_password or not confirm_new_password:
                    st.warning("Please fill in all password fields")
                elif not authenticate_user(st.session_state.username, current_password):
                    st.error("Current password is incorrect")
                elif new_password != confirm_new_password:
                    st.error("New passwords do not match")
                else:
                    valid_password, password_msg = validate_password(new_password)
                    if not valid_password:
                        st.error(password_msg)
                    else:
                        # Update password
                        with open(USER_DB_FILE, "r") as f:
                            users = json.load(f)
                        users[st.session_state.username] = hash_password(new_password)
                        with open(USER_DB_FILE, "w") as f:
                            json.dump(users, f)
                        
                        st.success("Password updated successfully!")
                        st.session_state.change_password = False
                        st.rerun()
        
        # Camera settings section
        st.subheader("Camera Settings")
        st.write("Configure camera behavior:")
        
        auto_stop_camera = st.checkbox("Automatically stop camera after analysis", 
                                      value=True if "auto_stop_camera" not in st.session_state else st.session_state.auto_stop_camera)
        
        if st.button("Save Camera Settings"):
            st.session_state.auto_stop_camera = auto_stop_camera
            st.success("Camera settings saved!")
    
    elif choice == "Logout":
        if st.button("Confirm Logout"):
            st.session_state.logged_in = False
            if "username" in st.session_state:
                del st.session_state.username
            st.success("You have been logged out")
            st.rerun()

# Main Streamlit app
def main():
    # Initialize database
    init_db()
    
    # Set page config
    st.set_page_config(
        page_title="AI Interview Coach",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if "show_login" not in st.session_state:
        st.session_state.show_login = True
        
    # Initialize camera active state by default
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = True
        
    # Initialize auto-stop camera preference
    if "auto_stop_camera" not in st.session_state:
        st.session_state.auto_stop_camera = True
    
    # Check if user is logged in
    if st.session_state.logged_in:
        main_app()
    else:
        st.title("AI-Powered Interview Coach")
        st.write("Improve your interview skills with personalized feedback on tone, word choice, and confidence.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Login" if not st.session_state.show_login else "Already have an account? Login"):
                st.session_state.show_login = True
                st.rerun()
        
        with col2:
            if st.button("Sign Up" if st.session_state.show_login else "Need an account? Sign Up"):
                st.session_state.show_login = False
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.show_login:
            create_login_page()
        else:
            create_signup_page()

if __name__ == "__main__":
    main()