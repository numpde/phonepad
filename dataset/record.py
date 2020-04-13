# RA, 2020-04-05

# Records audio and mouse at random times

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime as dt

from inclusive import range

import sounddevice as sd

PARAM = {
	'raw samples': Path(__file__).parent / "records/raw",

	# Number of channels to record (sounddevice library)
	'channels': 1,

	# Sample rate, aka frames/second (Hz)
	'fps': int(44100 / 2),

	# Audio record/sample duration (seconds)
	'duration': 2,

	# Preferred dtype for arrays
	'dtype': 'int16',
}

# Make sure the directory exists
PARAM['raw samples'].mkdir(exist_ok=True, parents=True)

# Classes / symbols to learn
# symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'tap', 'double-tap', 'back']
symbols = ['3', '4']


def record_to(filename):
	Path(filename).parent.mkdir(exist_ok=True, parents=True)

	sd.wait()

	audio = sd.rec(
		samplerate=PARAM['fps'],
		channels=PARAM['channels'],
		frames=round(PARAM['duration'] * PARAM['fps']),
		dtype=PARAM['dtype'],
	)

	sd.wait()

	# Time in seconds corresponding to audio frames
	tt = pd.Series(np.arange(audio.shape[0]) / PARAM['fps'], name="s")

	# Package audio as pandas dataframe with timestamps
	audio = pd.DataFrame(index=tt, data=audio)

	# Save to file
	audio.to_csv(filename, sep='\t', compression=dict(archive_name="audio.csv", method="zip"))

	# Recover the audio from file
	audio = pd.read_csv(filename, sep='\t', index_col=tt.name)

	return audio


for i in range[0, 10]:
	sample_id = dt.utcnow().strftime("%Y%m%d-%H%M%S-%f")

	if i:
		symbol = np.random.choice(symbols)
	else:
		symbol = "test"

	print(F"Record symbol `{symbol}` ", end="", flush=True)

	for p in "NOW":
		sd.sleep(500)
		print(p, end="", flush=True)

	print(".", end="", flush=True)
	audio = record_to(PARAM['raw samples'] / F"{symbol}/{sample_id}.zip")
	print("...OK")

	sd.sleep(500)

	sd.wait()
	sd.play(audio.to_numpy(dtype=PARAM['dtype']), samplerate=int(np.round(np.mean(1 / np.diff(audio.index)))))
	sd.wait()

	sd.sleep(500)
