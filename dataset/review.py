# RA, 2020-04-07

# Review the data samples

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sounddevice as sd

PARAM = {
	# Data samples
	'good samples': Path(__file__).parent / "records/raw",

	# Where to put low quality samples; must have
	'bad samples': Path(__file__).parent / "records/raw_bad",

	# Preferred dtype for arrays
	'dtype': 'int16',
}

samples = pd.DataFrame(
	data=[
		(str(f), f.stem, f.parent.name)
		for f in PARAM['good samples'].glob("*/*.zip")
	],
	columns=["file", "id", "label"],
).set_index("id", verify_integrity=True)

classes = set(samples.label)

# Select a class to review
while True:
	label_of_interest = input(F"Select a class ({classes}): ")
	if not label_of_interest:
		exit()
	if (label_of_interest in classes):
		break

# Focus on one class
samples = samples[samples.label == label_of_interest]

# Shuffle samples
samples.reindex(np.random.permutation(samples.index))

for (i, sample) in samples.iterrows():

	audio = pd.read_csv(sample.file, sep='\t', index_col=0)

	while True:
		print(F"Class '{sample.label}', sample ID {i}")

		sd.wait()
		sd.play(audio.to_numpy(dtype=PARAM['dtype']), samplerate=int(np.round(np.mean(1 / np.diff(audio.index)))))

		ok = input("Do you like it? ").lower().strip()

		if ok in ['y', 'n']:
			break

	if (ok == 'n'):
		print("Deleting...")
		sd.sleep(1)
	else:
		print("OK, keep it")
		sd.sleep(1)
		continue

	print("---")

	# Stash sample
	src = Path(sample.file)
	trg = Path(PARAM['bad samples'], src.relative_to(PARAM['good samples']))
	trg.parent.mkdir(exist_ok=True, parents=True)
	shutil.move(src, trg)

#
