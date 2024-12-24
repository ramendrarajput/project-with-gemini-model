import cv2
import os
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai
from google.cloud import texttospeech
import PIL.Image

def generate_frames_and_text(video, previous_texts, client):

    while True:
        success, frame = video.read()
        if not success:
            print("Failed to read the frame.")
            break

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Send the frame to Gemini and get the generated text
        generated_text = send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, client)
        print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")

        # Add the text with timestamp to the queue
        previous_texts.append(f"[{timestamp}] Message: {user_input}, Generated Text: {generated_text}")

        # Add text to the frame (Japanese characters may not display correctly)
        text_to_add = f"{timestamp}: {generated_text}" 

        add_text_to_frame(frame, text_to_add)

        # Save the frame
        filename = f"{timestamp}.jpg"
        save_frame(frame, filename)

        # Convert text to speech
        text_to_speech_google(generated_text, client)

        yield frame

def video_stream_generator(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        success, frame = video.read()
        if not success:
            print("Failed to read the frame.")
            break
        yield frame

    video.release()

def main():
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    # Initialize the Google Cloud TTS API client
    client = texttospeech.TextToSpeechClient()

    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("Could not open the camera.")
    except IOError as e:
        print(f"An error occurred: {e}")
        return

    # Queue to hold the text of the last 5 frames
    previous_texts = deque(maxlen=5)

    # Streamlit app
    import streamlit as st
    st.title("AI-Powered Video Chatbot")

    # Video stream
    video_generator = video_stream_generator("test.mp4")
    st.video(video_generator)

    # Text input
    user_input = st.text_input("Enter a new prompt or press Enter to continue (type 'exit' to end the program):")

    # Generate frames and text
    frames_and_text = generate_frames_and_text(video, previous_texts, client)

    # Display frames and text
    for frame in frames_and_text:
        st.image(frame)
        st.markdown(text_to_add)

    # Release the video
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()