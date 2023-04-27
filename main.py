from chat import Chat
from speak import Speak
from verify import Verify
import sys
import pyaudio
import numpy as np
import json
from vosk import Model, KaldiRecognizer, SpkModel
import queue
import threading
CHUNK = 8000
chat = Chat()
speak = Speak()
verify = Verify()
p = pyaudio.PyAudio()

# 加载Vosk模型
model = Model("model")
spk_model = SpkModel("spk")

# 初始化语音识别器
rec = KaldiRecognizer(model, 16000)
rec.SetSpkModel(spk_model)
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNK)
spk_sig = [0.744, 0.470776, 0.409491, 1.072628, -0.009881, -0.39243, 1.172459, -0.930088, 1.3822, -0.203496, 0.514394, -1.270681, -0.990526, 0.432183, 0.507454, 1.62457, -0.476533, 1.956429, -0.318385, 0.264149, -2.413229, 0.465301, 0.790405, -1.489686, 0.509152, -0.225385, -0.832558, 0.524678, -1.896292, -0.698305, -0.603268, -0.105941, -2.054857, -0.207751, 2.268001, 0.885217, 1.081729, 0.869747, -2.460924, -0.924878, -0.013494, -0.988696, 2.308232, 0.682755, 0.149429, 1.587425, 0.982061, 0.447203, -0.447687, -1.831222, 0.058419, 1.523724, -1.37929, -0.541588, -0.820471, -1.071049, 0.207722, -0.611727, -0.933272, -0.381253, 1.465553, 1.152384, -0.380897, -0.282967, -2.387321, -0.124108, -0.533252, 0.655543, -1.070736, 0.468435, 1.329845, 1.300223, -0.365832, 1.020658, -0.787601, 0.509791, 0.178922, -0.948506, 0.548219, -0.412294, 0.423733, -1.415372, -0.763544, -0.257024, 0.022973, 0.045699, 0.770287, 1.032928, -1.65747, -0.158873, -1.082766, 0.175328, -1.284591, 0.134394, 0.909849, -1.136633, 0.265265, 0.625835, 0.415473, 1.187992, -0.195717, -1.468426, 2.386842, 0.377205, 0.501866, 0.826397, 0.112176, -0.517755, 1.098556, -0.060629, 0.474657, 0.767659, 2.014373, 0.825794, 0.602801, 0.102287, -0.91932, 0.343257, 0.279488, 0.620493, -0.750622, -0.63066, -0.584985, -0.183122, 0.37886, 0.414863, -1.853542, 0.55881]
q = queue.Queue()
messages = []
rec.SetWords(True)

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))



def handle_message(res):
	print(res)
	if verify.is_speaker(res["spk"]) > 0.5:
		print("request chat gpt...")
		answer = chat.ask(res["text"])
		speak.say(answer)
	

def read_stream():
    while True:
        data = stream.read(CHUNK)
        q.put(data)
read_thread = threading.Thread(target=read_stream)
read_thread.start()

def main():
        # 初始化对话列表，可以加入一个key为system的字典，有助于形成更加个性化的回答
        # self.conversation_list = [{'role':'system','content':'你是一个非常友善的助手'}]
        try:
        	print("start")
        	while True:
		        data = q.get()
		        if rec.AcceptWaveform(data):
		            res = json.loads(rec.Result())
		            if len(res["text"]) > 0:
		            	handle_message(res)
        except KeyboardInterrupt:
        	exit(0)
        except Exception as e:
        	exit(type(e).__name__ + ": " + str(e))
if __name__ == '__main__':
    main()
