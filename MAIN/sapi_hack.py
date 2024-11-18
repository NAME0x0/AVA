import win32com.client

def list_voices():
    sapi = win32com.client.Dispatch("SAPI.SpVoice")
    voices = sapi.GetVoices()
    print("Available voices:")
    for i, voice in enumerate(voices):
        print(f"{i+1}. {voice.GetDescription()}")

def select_voice():
    sapi = win32com.client.Dispatch("SAPI.SpVoice")
    voices = sapi.GetVoices()
    while True:
        try:
            voice_index = int(input("Enter the number of the voice you want to use: "))
            if 1 <= voice_index <= len(voices):
                return voices[voice_index - 1]
            else:
                print("Invalid voice number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    list_voices()
    selected_voice = select_voice()
    
    sapi = win32com.client.Dispatch("SAPI.SpVoice")
    sapi.Voice = selected_voice

    text = "Hello, this is a test using the selected voice."
    sapi.Speak(text)

if __name__ == "__main__":
    main()
