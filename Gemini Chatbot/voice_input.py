import speechrecognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except:
        print("Sorry, I didn't catch that.")
        return None

# Main loop
while True:
    text = speech_to_text()
    if text:
        # Process the text input here
        response = process_text(text)
        print(response)