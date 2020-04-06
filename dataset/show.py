# RA, 2020-04-06

# Visualize recodings

import re
import pandas as pd
import numpy as np
from pathlib import Path
from more_itertools import first

PARAM = {
	'samples path': Path(__file__).parent / "records/raw",
}

samples = pd.DataFrame(
	data=[
		(str(f), first(f.name.split('.')), first(f.suffixes)[1:])
		for f in PARAM['samples path'].glob("*.zip")
	],
	columns=["file", "id", "type"],
).pivot_table(
	values="file", index="id", columns="type",
	aggfunc=(lambda df: df.squeeze())
)

# Select one sample
row = samples.iloc[1]

audio = pd.read_csv(row['audio'], sep='\t', index_col="s")
mouse = pd.read_csv(row['mouse'], sep='\t', index_col="s")

# Frames per second (framerate)
fps = int(1 / np.mean(np.diff(audio.index)))

from scipy.interpolate import interp1d
# from scipy.stats import gaussian_kde
# gaussian_kde()
mouse = pd.DataFrame(
	data=interp1d(x=mouse.index, y=mouse.T)(audio.index).T,
	index=audio.index,
	columns=mouse.columns
).diff().rolling(round(0.1 * fps)).mean()

mouse = mouse.drop(columns='y')

# import matplotlib.pyplot as plt
# fig: plt.Figure
# ax1: plt.Axes
# ax2: plt.Axes
# (fig, ax1) = plt.subplots()
# ax2 = ax1.twinx()
# # ax2.plot(audio.index, audio.to_numpy(), c="C3")
# print()
# ax1.specgram(audio.to_numpy().squeeze(), Fs=fps, mode="psd")
# ax2.plot(mouse.index, mouse.to_numpy())
# plt.show()
# exit()


from scipy.signal import spectrogram
(ff, tt, sg) = spectrogram(audio.to_numpy().squeeze(), fs=fps, nperseg=int(0.1 * fps), mode='psd')
ff_ok = (10 < ff) & (ff < 2000)
sg = sg[ff_ok, :]
# print(ff.shape, tt.shape, sg.shape)
# print(tt)
y = pd.DataFrame(index=list(tt), data=interp1d(x=mouse.index, y=mouse.T)(tt).T)
X = pd.DataFrame(index=list(tt), data=sg.T, columns=ff[ff_ok])
y = y.dropna()
X = X.loc[y.index]

ii_train = y.index[np.random.choice([True, False], size=len(y.index))]
ii_valid = y.index.difference(ii_train)

from sklearn.linear_model import Lasso as Regressor
model = Regressor(alpha=1e-3, fit_intercept=True, normalize=False).fit(X=X.loc[ii_train], y=y.loc[ii_train])
z = pd.DataFrame(index=X.index, data=model.predict(X))


import matplotlib.pyplot as plt
fig: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
(fig, ax1) = plt.subplots()
ax2 = ax1.twinx()
# ax2.plot(audio.index, audio.to_numpy(), c="C3")
# ax1.plot(y.loc[ii_valid], c="C1")
# ax2.plot(z.loc[ii_valid], c="C2")
ax1.scatter(y.loc[ii_train], z.loc[ii_train])
ax1.scatter(y.loc[ii_valid], z.loc[ii_valid])
ax1.grid()
plt.show()

# import matplotlib.pyplot as plt
# plt.plot(y.rolling(30).corr(z))
# plt.plot(y.rolling(50).corr(z))
# plt.plot(y.rolling(70).corr(z))
# plt.show()