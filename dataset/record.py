# RA, 2020-04-05

# Records audio and mouse at random times

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime as dt

import sounddevice as sd
import pyautogui


PARAM = {
	'raw samples': Path(__file__).parent / "records/raw",

	# Number of channels to record (sounddevice library)
	'channels': 1,

	# Sample rate, aka frames/second (Hz)
	'fps': int(44100 / 2),

	# Audio record/sample duration (seconds)
	'duration': 30,

	# Preferred dtype for arrays
	'dtype': 'int16',
}

# Make sure the directory exists
PARAM['raw samples'].mkdir(exist_ok=True, parents=True)


sample_id = dt.utcnow().strftime("%Y%m%d-%H%M%S-%f")
print("Recording sample ID:", sample_id)


mouse = pd.DataFrame(columns=["x", "y"], dtype=PARAM['dtype'])

audio = sd.rec(
	frames=round(PARAM['duration'] * PARAM['fps']),
	samplerate=PARAM['fps'],
	channels=PARAM['channels'],
	dtype=PARAM['dtype']
)

while sd.get_stream().active:
	mouse.loc[dt.utcnow().timestamp()] = pyautogui.position()

# Make index start from zero
mouse = mouse.set_index(pd.Index(mouse.index - mouse.index[0], name="s"))

# Time in seconds corresponding to audio frames
tt = pd.Series(np.arange(audio.shape[0]) / PARAM['fps'], name="s")

# Package audio as pandas dataframe with timestamps
audio = pd.DataFrame(index=tt, data=audio)

# print(mouse)
# print(audio)

# import matplotlib.pyplot as plt
# plt.plot(audio)
# plt.show()

filename = str(PARAM['raw samples'] / F"{sample_id}.{{ext}}")
mouse.to_csv(filename.format(ext="mouse.zip"), sep='\t', compression=dict(archive_name="mouse.csv", method="zip"))
audio.to_csv(filename.format(ext="audio.zip"), sep='\t', compression=dict(archive_name="audio.csv", method="zip"))

# from scipy.io.wavfile import write
# write(filename.format(ext="audio.wav"), fs, audio.to_numpy())

# sd.play(audio.to_numpy(), samplerate=fs)
# sd.wait()
