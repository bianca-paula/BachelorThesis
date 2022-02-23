# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime

import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os

#Build the Chatbot
from playsound import playsound


class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me --> ERROR")

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("result.mp3")
        playsound("result.mp3")
        #os.system("start res.mp3")
        os.remove("result.mp3")

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')



    def wake_up(self, text):
        return True if self.name in text.lower() else False

# Press the green button in the gutter to run the script.
#Run the ChatBot
if __name__ == '__main__':
    ai = ChatBot(name="maya")
    while True:
        ai.speech_to_text()
        ## wake up
        if ai.wake_up(ai.text) is True:
            res="Hello I am Maya the AI, what can I do for you?"
        ## action time
        elif "time" in ai.text:
            res=ai.action_time()
        ##respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res=np.random.choice(
                ["You're welcome!",
                 "Anytime",
                 "No problem!",
                 "Cool!",
                 "I'm here if you need me!"
                ]
            )
        ai.text_to_speech(res)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
