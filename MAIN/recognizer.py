import whisper

def setup_recognizer():
    recognizer = whisper.Recognizer()
    return recognizer

def load_audio_model():
    model = whisper.load("base.en")  # Load the Whisper model for English
    return model

def process_command(user_input, recognizer, audio_model):
    try:
        with whisper.Microphone() as source:
            print("Listening...:", user_input)
            audio = recognizer.listen(source, timeout=5)
        command = recognizer.recognize_whisper(audio, model=audio_model)
        print("User said:", command)
        # Use female voice for speech synthesis
        whisper.speak(command, voice="female")
        return command
    except whisper.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except whisper.RequestError as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Error: {e}"
