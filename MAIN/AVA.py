import asyncio
import datetime
import math
import pyautogui
import psutil
import pyttsx3
import pandas as pd
import speech_recognition as sr
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from audio import Audio  # Import the Audio class from audio.py
from chat import Chat  # Import the Chat class from chat.py

class AVA:
    def __init__(self, wake_up_command="Hey AVA", name="Afsah"):
        self.name = name
        self.volume = self.get_default_audio_endpoint_volume()
        self.engine = pyttsx3.init()
        self.audio = Audio(engine=self.engine)  # Pass engine object
        self.wake_up_command = wake_up_command.lower()
        self.recognizer = sr.Recognizer()
        self.greet()

    def greet(self):
        current_time = datetime.datetime.now().time()
        if datetime.time(5, 0) <= current_time < datetime.time(12, 0):
            greet_message = f"Good morning, {self.name}! How can I help you right now?"
            print(greet_message)
            self.audio.speak(greet_message)
        elif datetime.time(12, 0) <= current_time < datetime.time(17, 0):
            greet_message = f"Good afternoon, {self.name}! What can I do to help you?"
            print(greet_message)
            self.audio.speak(greet_message)
        else:
            greet_message = f"Hi, {self.name}! What do you need help with?"
            print(greet_message)
            self.audio.speak(greet_message)

    def change_voice(self, command):
        try:
            if command.lower() == "change voice":
                available_voices = self.audio.list_voices()  # List available voices
                print("Available voices:")
                for voice in available_voices:
                    print(voice)
                return "Available voices:"
            elif command.lower().startswith("change voice to"):
                voice_choice = command.split(" ")[-1]  # Extract the voice choice from the command
                try:
                    voice_index = int(voice_choice) - 1  # Convert to zero-based index
                    self.audio.set_voice_by_index(voice_index)  # Set the selected voice
                    return f"Voice changed to {voice_choice}"
                except ValueError:
                    return "Invalid voice choice. Please provide a valid number."
            else:
                return "Unknown command. Please use 'change voice' to list available voices or 'change voice to <number>' to change the voice."
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)
            return error_message

    def process_command_with_speech(self):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.audio.speak("Listening...")  # Speak "Listening..."
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.listen(source, timeout=5)
            command = self.recognizer.recognize_google(audio)
            print("You: ", command)
            self.audio.speak("You said " + command)  # Speak the recognized command
            return command
        except sr.RequestError as e:
            return f"Could not request results; {e}"
            self.audio.speak("I am so sorry but I could not request the results for your command!")
        except sr.UnknownValueError:
            return "Unknown error occurred"
            self.audio.speak("An unknown error occurred. I apologize for the inconvenience")

    async def process_command(self, user_input):
        if user_input.lower() == "exit":
            return f"Goodbye {user_name}. Have a nice day!"
        elif any(greeting in user_input.lower() for greeting in ["hi", "hey"]):
            return self.greet()
        elif "time" in user_input.lower():
            return self.get_current_time()
        elif "date" in user_input.lower():
            return self.get_current_date()
        elif user_input.lower() == "pi":
            return math.pi
        elif "/" in user_input:
            return self.perform_math_operation(user_input, "/")
        elif "*" in user_input:
            return self.perform_math_operation(user_input, "*")
        elif "+" in user_input:
            return self.perform_math_operation(user_input, "+")
        elif "-" in user_input:
            return self.perform_math_operation(user_input, "-")
        elif any(keyword in user_input.lower() for keyword in ["volume", "adjust"]):
            return self.adjust_volume(user_input)
        elif "play" in user_input.lower():
            pyautogui.press("playpause")
            return "Media playback resumed"
        elif "pause" in user_input.lower():
            pyautogui.press("playpause")
            return "Media playback paused"
        elif user_input.lower().startswith("close app"):
            return self.list_and_close_app()
        elif "change voice" in user_input.lower():
            return self.change_voice(user_input)
        elif self.wake_up_command in user_input.lower():  # Check if the wake-up command is in the user input
            return "Yes?"

    def get_current_time(self):
        try:
            current_time = datetime.datetime.now().strftime("%H:%M")
            time_response = f"The current time is: {current_time}"
            print(time_response)
            self.audio.speak(time_response)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def get_current_date(self):
        try:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            date_response = f"The current date is: {current_date}"
            print(date_response)
            self.audio.speak(date_response)
            return date_response
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)            

    def adjust_volume(self, user_input):
        try:
            if "current" in user_input.lower():
                current_volume = f"The current volume level is: {self.get_current_volume()}%"
                print(current_volume)
                self.audio.speak(current_volume)
            else:
                # Extract the desired volume level from the user input
                volume_level = int(''.join(filter(str.isdigit, user_input)))
                self.set_volume(volume_level)
                return f"Volume level set to {volume_level}%"
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def set_volume(self, volume_level):
        try:
            self.volume.SetMasterVolumeLevelScalar(volume_level / 100, None)
            volume_adjustment_response = "Volume adjusted."
            print(volume_adjustment_response)
            self.audio.speak(volume_adjustment_response)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def get_current_volume(self):
        try:
            current_volume_response = int(self.volume.GetMasterVolumeLevelScalar() * 100)
            print(current_volume_response)
            self.audio.speak(current_volume_response)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def perform_math_operation(self, user_input, operator):
        try:
            operands = user_input.split(operator)
            result = eval(operands[0].strip() + operator + operands[1].strip())
            return str(result)
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)
        
    def get_default_audio_endpoint_volume(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def list_and_close_app(self):
        try:
            excluded_processes = ["System Idle Process", "System", "Registry", "svchost.exe", "wininit.exe", "winlogon.exe", "csrss.exe", "smss.exe", "lsass.exe", "explorer.exe"]
            running_apps = []
            pids = []

            for proc in psutil.process_iter(['pid', 'name']):
                process_name = proc.info['name']
                process_pid = proc.info['pid']
                if process_name not in excluded_processes:
                    running_apps.append(process_name)
                    pids.append(process_pid)

            if not running_apps:
                return "No active apps found."

            # Create a DataFrame with app names and PIDs
            df = pd.DataFrame({"App Name": running_apps, "PID": pids})

            # Sort apps alphabetically by name
            df_sorted = df.sort_values(by='App Name', key=lambda x: x.str.lower())

            print(f"Apps currently running:\n{df_sorted.to_string(index=False)}")

            # Prompt the user using TTS
            self.audio.speak("Apps currently running. Which would you like me to close? Type 'none' to cancel.")

            while True:
                app_name = input("You: ").strip()
                if app_name.lower() == 'none':
                    return "Operation canceled."
                else:
                    filtered_df = df_sorted[df_sorted['App Name'].str.lower() == app_name.lower()]
                    if not filtered_df.empty:
                        # Get all PIDs corresponding to the app name
                        pids_to_close = filtered_df['PID'].tolist()
                        # Close all processes with the given app name
                        for pid_to_close in pids_to_close:
                            for proc in psutil.process_iter(['pid', 'name']):
                                if proc.info['pid'] == pid_to_close and proc.info['name'] == app_name:
                                    proc.kill()
                        return f"Closed {app_name}"
                    else:
                        print(f"App '{app_name}' not found or not running. Please choose from the listed apps.")
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            self.audio.speak(error_message)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == "__main__":
    user_name = "Afsah"
    ava = AVA(name=user_name)
    while True:
        user_input = input("You: ")
        response = asyncio.run(ava.process_command(user_input))
        if response:
            ava.audio.speak(response)  # Speak the response using TTS
            print("AVA:", response)
            if response.startswith("Available voices:"):
                # List available voices
                print(response)
            elif response.startswith("Voice changed to"):
                # Voice change successful
                print(response)
            elif response.startswith(f"Goodbye {user_name}. Have a nice day!"):
                break
            elif response.lower() == "change voice":
                # Ask for available voices
                response = ava.change_voice(response)
                print(response)
            else:
                # Process other commands
                if response == "change voice":
                    response = ava.change_voice(response)
                elif response.startswith(f"Goodbye {user_name}. Have a nice day!"):
                    break
