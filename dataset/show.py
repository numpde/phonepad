# RA, 2020-04-06

# Visualize recordings

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sounddevice as sd

PARAM = {
	'samples path': Path(__file__).parent / "records/raw",

	# Preferred dtype for audio arrays
	'dtype': 'int16',
}

samples = pd.DataFrame(
	data=[
		(str(f), f.stem, f.parent.name)
		for f in PARAM['samples path'].glob("*/*.zip")
	],
	columns=["file", "id", "label"],
).set_index("id", verify_integrity=True)

# Hear/see one sample
print(samples)

while True:
	print()
	i = input("Which sample would you like to see/hear? ")
	try:
		sample = samples.loc[i]
	except:
		continue
		# sample = samples.sample(1).squeeze()

	print("Sample:")
	print(pd.DataFrame(sample).T.to_markdown())


	# Load the audio
	audio = pd.read_csv(sample['file'], sep='\t', index_col='s')

	# Frames per second (framerate)
	fps = int(np.round(np.mean(1 / np.diff(audio.index))))

	# Hit 'em with the graph

	fig: plt.Figure
	ax1: plt.Axes
	ax2: plt.Axes
	(fig, ax1) = plt.subplots()

	fig.suptitle(F"ID: {sample.name}, class label: '{sample.label}'")

	ax1.specgram(x=audio.squeeze(), Fs=fps, mode="psd")
	ax1.set_xlabel("Time (seconds)")
	ax1.set_ylabel("Frequency (Hz)")

	(ff, tt, spectrum) = spectrogram(audio.squeeze(), fs=fps, nperseg=int(0.05 * fps), mode='psd')
	freq_cutoff = 400
	spectrum = spectrum[(freq_cutoff <= ff), :]
	ax2 = ax1.twinx()
	ax2.plot(tt, np.sum(spectrum, axis=0), c="white")
	ax2.set_ylabel(F"Power above {freq_cutoff} Hz")
	ax2.set_yticks([])

	sd.wait()
	sd.play(audio.to_numpy(dtype=PARAM['dtype']), samplerate=fps)

	plt.show()

