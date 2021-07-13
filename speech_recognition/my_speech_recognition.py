"""
https://realpython.com/python-speech-recognition/#how-speech-recognition-works-an-overview

"""
# SpeechRecognition 
# See: https://pypi.org/project/SpeechRecognition/
import speech_recognition as sr 

r = sr.Recognizer() # create a recognizer instance



mic = sr.Microphone()

sr.Microphone.list_microphone_names()

# mic = sr.Microphone(device_index=3) # For example

with mic as source:
	audio = r.listen(source)


# Example of running a web file
"""
file1 = 'harvard.wav'
file2 = 'jackhammer.wav'
start = 0
end = 4
harvard = sr.AudioFile(file='harvard.wav', offset=start, duration=end)
with harvard as source:
	audio = r.record(source)

r.recognize_google(audio)


r.recognizer_google(audio, show_all=True)
"""


