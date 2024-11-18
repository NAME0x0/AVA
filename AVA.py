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

class AVA:
    def __init__(self, wake_up_command="Hey AVA"):
        self.name = "Afsah"
        self.volume = self.get_default_audio_endpoint_volume()
        self.audio = Audio()  # Create an instance of the Audio class for TTS
        self.wake_up_command = wake_up_command.lower()
        self.recognizer = sr.Recognizer()

    def greet(self):
        current_time = datetime.datetime.now().time()
        if datetime.time(5, 0) <= current_time < datetime.time(12, 0):
            return "Good morning, Afsah! How can I help you right now?"
        elif datetime.time(12, 0) <= current_time < datetime.time(17, 0):
            return "Good afternoon, Afsah! What can I do to help you?"
        else:
            return "Hi, Afsah! What do you need help with?"

    async def process_command(self, user_input):
        if user_input.lower() == "exit":
            return "Goodbye!"
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
        else self.wake_up_command in user_input.lower():  # Check if the wake-up command is in the user input
            return "Yes?"

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

    def get_current_time(self):
        try:
            current_time = datetime.datetime.now().strftime("%H:%M")
            return f"The current time is: {current_time}"
        except Exception as e:
            return f"Error: {e}"

    def get_current_date(self):
        try:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            return f"The current date is: {current_date}"
        except Exception as e:
            return f"Error: {e}"

    def adjust_volume(self, user_input):
        try:
            if "current" in user_input.lower():
                return f"The current volume level is: {self.get_current_volume()}%"
            else:
                # Extract the desired volume level from the user input
                volume_level = int(''.join(filter(str.isdigit, user_input)))
                self.set_volume(volume_level)
                return f"Volume level set to {volume_level}%"
        except Exception as e:
            return f"Error: {e}"

    def set_volume(self, volume_level):
        try:
            self.volume.SetMasterVolumeLevelScalar(volume_level / 100, None)
            return "Volume adjusted."
        except Exception as e:
            return f"Error: {e}"

    def get_current_volume(self):
        try:
            return int(self.volume.GetMasterVolumeLevelScalar() * 100)
        except Exception as e:
            return f"Error: {e}"

    def perform_math_operation(self, user_input, operator):
        try:
            operands = user_input.split(operator)
            result = eval(operands[0].strip() + operator + operands[1].strip())
            return result
        except Exception as e:
            return f"Error: {e}"
        
    def get_default_audio_endpoint_volume(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            return f"Error: {e}"

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

            print(f"Apps currently running:\n{df_sorted}")

            while True:
                app_name = input("Which app would you like to close? (Type 'none' to cancel): ").strip()
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
            return f"Error: {e}"

if __name__ == "__main__":
    ava = AVA()
    print(ava.greet())  # Print initial greeting when AVA is booted up
    while True:
        user_input = input("You: ")
        response = asyncio.run(ava.process_command(user_input))
        ava.audio.speak(response)  # Speak the response using TTS
        print("AVA:", response)
        if response == "Goodbye!":
            break
