import pickle
import os
import numpy as np

path = "speaker.bin"
class Verify:
    def __init__(self):
        self.thresold = 0.5
        if not os.path.exists(path):
            self.gen()
        else:
            with open(path, "rb") as f:
                serialized_data = f.read()
            data = pickle.loads(serialized_data)
            self.spk_sig = data["spk_sig"]
            self.times = data["times"]
        print("times %d" %self.times)

    def write(self):
        data = {
            "spk_sig": self.spk_sig,
            "times": self.times
        }
        with open(path, "wb") as f:
            f.write(pickle.dumps(data))

    def gen(self):
        self.times = 1
        self.spk_sig = [0.744, 0.470776, 0.409491, 1.072628, -0.009881, -0.39243, 1.172459, -0.930088, 1.3822, -0.203496, 0.514394, -1.270681, -0.990526, 0.432183, 0.507454, 1.62457, -0.476533, 1.956429, -0.318385, 0.264149, -2.413229, 0.465301, 0.790405, -1.489686, 0.509152, -0.225385, -0.832558, 0.524678, -1.896292, -0.698305, -0.603268, -0.105941, -2.054857, -0.207751, 2.268001, 0.885217, 1.081729, 0.869747, -2.460924, -0.924878, -0.013494, -0.988696, 2.308232, 0.682755, 0.149429, 1.587425, 0.982061, 0.447203, -0.447687, -1.831222, 0.058419, 1.523724, -1.37929, -0.541588, -0.820471, -1.071049, 0.207722, -0.611727, -0.933272, -0.381253, 1.465553, 1.152384, -0.380897, -0.282967, -2.387321, -0.124108, -0.533252, 0.655543, -1.070736, 0.468435, 1.329845, 1.300223, -0.365832, 1.020658, -0.787601, 0.509791, 0.178922, -0.948506, 0.548219, -0.412294, 0.423733, -1.415372, -0.763544, -0.257024, 0.022973, 0.045699, 0.770287, 1.032928, -1.65747, -0.158873, -1.082766, 0.175328, -1.284591, 0.134394, 0.909849, -1.136633, 0.265265, 0.625835, 0.415473, 1.187992, -0.195717, -1.468426, 2.386842, 0.377205, 0.501866, 0.826397, 0.112176, -0.517755, 1.098556, -0.060629, 0.474657, 0.767659, 2.014373, 0.825794, 0.602801, 0.102287, -0.91932, 0.343257, 0.279488, 0.620493, -0.750622, -0.63066, -0.584985, -0.183122, 0.37886, 0.414863, -1.853542, 0.55881]
        self.write()

    def is_speaker(self, spk):
        dist = self.cosine_dist(self.spk_sig, spk)
        print(dist)
        if dist > self.thresold:
            return True
        else:
            self.append_new_sig(spk)
            return False

    def append_new_sig(self, spk):
        self.spk_sig = np.average([self.spk_sig, spk], axis=0, weights=[self.times, 1])
        self.times += 1
        self.write()

    def cosine_dist(self, x, y):
        nx = np.array(x)
        ny = np.array(y)
        return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
