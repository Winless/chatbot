from AppKit import NSSpeechSynthesizer

class Speak:
    def __init__(self):
        self.synth = NSSpeechSynthesizer.new()

    def say(self, message):
        self.synth.startSpeakingString_(message)
        while self.synth.isSpeaking():
            pass