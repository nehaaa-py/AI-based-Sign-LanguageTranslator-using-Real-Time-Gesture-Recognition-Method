import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

engine.say("Hello  Neha Sharma, your sign language translator project is working.")
engine.runAndWait()


