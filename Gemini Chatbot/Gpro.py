import cv2
import os
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai
from google.cloud import texttospeech
import PIL.Image

def text_to_speech_google(text, client):
    # Setting up the text-to-speech request
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Sending the text-to-speech request
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Saving the speech data to a file
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    # Loading the MP3 file
    sound = AudioSegment.from_mp3("output.mp3")
    # Playing the sound
    play(sound)

def wrap_text(text, line_length):
    """Wrap the text at the specified length"""
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word

    lines.append(current_line)  # Add the last line
    return lines

def add_text_to_frame(frame, text):
    # Wrap the text every 70 characters
    wrapped_text = wrap_text(text, 70)

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Setting the font and size for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Increase font size
    color = (255, 255, 255)  # White color
    outline_color = (0, 0, 0)  # Outline color (black)
    thickness = 2
    outline_thickness = 4  # Thickness of the outline
    line_type = cv2.LINE_AA

    # Adding each line of text to the image
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 30)  # Adjust the position of each line (larger interval)

        # Drawing the outline of the text
        cv2.putText(frame, line, position, font, font_scale, outline_color, outline_thickness, line_type)

        # Drawing the text
        cv2.putText(frame, line, position, font, font_scale, color, thickness, line_type)

def save_frame(frame, filename, directory='./frames'):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Create the path for the filename
    filepath = os.path.join(directory, filename)
    # Save the frame
    cv2.imwrite(filepath, frame)

def save_temp_frame(frame, filename, directory='./temp'):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Create the path for the filename
    filepath = os.path.join(directory, filename)
    # Save the frame
    cv2.imwrite(filepath, frame)
    return filepath  # Return the path of the saved file

def send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, client):
    
    temp_file_path = save_temp_frame(frame, "temp.jpg")
    img = PIL.Image.open(temp_file_path)

    # Combine past texts as context
    context = ' '.join(previous_texts)

    # Initialize the Gemini model
    model = client.GenerativeModel('gemini-pro-vision')

    # Send image and text instructions to the model
    prompt = f"Given the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context. Message: {user_input}"
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    # Return the generated text
    return response.text

def main():
    
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
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

    ##################################################################################


    ##################################################################################

    while True:
        
        print("Enter a new prompt or press Enter to continue (type 'exit' to end the program):")
        user_input = input().strip()  # Receive input

        if not user_input:
            user_input = "Tell me what you see."

        success, frame = video.read()
        if not success:
            print("Failed to read the frame.")
            break

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Send the frame to Gemini and get the generated text
        generated_text = send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, genai)
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

    # Release the video
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()