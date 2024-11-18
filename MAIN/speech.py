import pyttsx3

engine = pyttsx3.init()

voice_num = 2
text_to_say = "Hello Afsah! I am AVA!"

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[voice_num].id)

engine.say(text_to_say)
engine.runAndWait()
