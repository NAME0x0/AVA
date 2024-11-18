import asyncio
import datetime
import math
import keyboard
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class AVA:
    def __init__(self):
        self.name = "Afsah"
        self.volume = self.get_default_audio_endpoint_volume()

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
        elif any(keyword in user_input.lower() for keyword in ["play", "pause"]):
            return self.control_media(user_input)
        else:
            return "Unknown command."

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

    def control_media(self, user_input):
        try:
            if "play" in user_input.lower():
                pyautogui.press('playpause')
                return "Media playback resumed."
            elif "pause" in user_input.lower():
                pyautogui.press('playpause')
                return "Media playback paused."
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    ava = AVA()
    print(ava.greet())  # Print initial greeting when AVA is booted up
    while True:
        user_input = input("You: ")
        response = asyncio.run(ava.process_command(user_input))
        print("AVA:", response)
        if response == "Goodbye!":
            break