import pyttsx3
import os
import winreg

class Audio:
    def __init__(self, engine):
        self.engine = engine
        self.voices = self.engine.getProperty('voices')
        self.registry_voices = self._get_registry_voices()
        self.sonias_voice_name = self._get_sonia_voice()
        self.available_voices = self.registry_voices + [self.sonias_voice_name] + [voice.name for voice in self.voices]

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def _get_registry_voices(self):
        try:
            registry_path = r"SOFTWARE\Microsoft\Speech\Voices\Tokens"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
                index = 0
                registry_voices = []
                while True:
                    try:
                        sub_key_name = winreg.EnumKey(key, index)
                        with winreg.OpenKey(key, sub_key_name) as sub_key:
                            value, _ = winreg.QueryValueEx(sub_key, "Name")
                            registry_voices.append(value)
                        index += 1
                    except OSError:
                        break
                return registry_voices
        except Exception as e:
            print(f"Error accessing registry: {e}")
            return []

    def _get_sonia_voice(self):
        sonia_voice_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sonia")
        if os.path.exists(sonia_voice_path):
            return f"Sonia voice ({os.path.basename(sonia_voice_path)})"
        return ""

    def list_voices(self):
        available_voices = []
        for i, voice in enumerate(self.available_voices):
            available_voices.append(f"{i+1}. {voice}")
        return available_voices

    def set_voice_by_index(self, index):
        if 0 <= index < len(self.available_voices):
            if index < len(self.registry_voices):
                selected_voice = self.registry_voices[index]
                self.engine.setProperty('voice', selected_voice)
                print(f"Voice set to: {selected_voice}")
            elif index == len(self.registry_voices):
                self.engine.setProperty('voice', self.sonias_voice_name)
                print(f"Voice set to: {self.sonias_voice_name}")
            else:
                self.engine.setProperty('voice', self.voices[index - len(self.registry_voices) - 1].id)
                print(f"Voice set to: {self.voices[index - len(self.registry_voices) - 1].name}")
        else:
            print("Invalid voice index.")
