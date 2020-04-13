# RA, 2020-04-06

import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf

# Setup logger
import logging as logger

logger.basicConfig(level=logger.DEBUG)

# tf.config.set_visible_devices([], 'GPU')
# tf.get_logger().setLevel(logger.ERROR)

from inclusive import range

import pandas as pd
import numpy as np

from pathlib import Path
from scipy.signal import spectrogram

from collections import Counter

from datetime import datetime as dt

PARAM = {
	'samples path': Path(__file__).parent.parent / "dataset/records/raw",

	'do spectrogram': True,

	'log dir': Path(__file__).parent / ("logs/" + dt.utcnow().strftime("%Y%m%d-%H%M%S-%f"))
}

import shutil
PARAM['log dir'].mkdir(parents=True, exist_ok=True)
shutil.copy(Path(__file__), PARAM['log dir'] / Path(__file__).name)

samples = pd.DataFrame(
	data=[
		(str(f), f.stem, f.parent.name)
		for f in PARAM['samples path'].glob("*/*.zip")
	],
	columns=["file", "id", "label"],
).set_index("id", verify_integrity=True)

samples = samples[samples['label'].isin(['3', '4'])]

# Class labels
classes = sorted(set(samples['label']))
nsamples = len(samples)
batch_size = 8
shuffle_buffer = 30
shuffle_seed = 111

# Features / timestep
nfeatures = {True: 221, False: 1}[PARAM['do spectrogram']]

validation_split = 0.25
samples_valid = samples.sample(n=round(validation_split * len(samples)))
samples_train = samples[~samples.index.isin(samples_valid.index)]

logger.info(F"Number of train/valid samples: {len(samples_train)}/{len(samples_valid)}")

del samples


def get_fps(sample):
	return int(np.round(np.mean(1 / np.diff(sample.index))))


def load_audio(file):
	audio = pd.read_csv(file, sep='\t', index_col=0)

	if PARAM['do spectrogram']:
		nperseg = int(0.02 * get_fps(audio))
		assert (nperseg == 441)
		# Note: nperseg determines the frequency lumping in the spectrogram (and the range a little)
		(ff, tt, sg) = spectrogram(audio.squeeze(), fs=get_fps(audio), nperseg=nperseg)
		assert (1e-6 > abs(min(ff)))
		assert (1e-6 > abs(max(ff) - 11000))
		assert (len(ff) == 221)
		# logger.info((nperseg, audio.shape, get_fps(audio), int(0.02 * get_fps(audio)),
		# ff.shape, tt.shape, sg.shape, [min(ff), max(ff)], [min(tt), max(tt)]))
		return sg.T
	else:
		return audio.to_numpy()


class Generator:
	def __init__(self, samples, with_id=False):
		self.samples = samples
		self.with_id = with_id

	def __len__(self):
		return len(self.samples)

	def __call__(self, *args, **kwargs):
		return self

	def __iter__(self):
		self.i = iter(self.samples.iterrows())
		return self

	def __next__(self):
		(i, sample) = next(self.i)
		(x, y) = (load_audio(sample.file), classes.index(sample.label))
		if self.with_id:
			return (x, [y], [i])
		else:
			return (x, [y])


def dataset_from_samples(samples, with_id=False):
	ds = tf.data.Dataset.from_generator(
		Generator(samples, with_id),
		output_types=(tf.float32, tf.int32, tf.string)[0:(2 + with_id)],
		output_shapes=(tf.TensorShape((None, nfeatures)), tf.TensorShape((1,)), tf.TensorShape((1,)))[0:(2 + with_id)]
	).prefetch(10)

	# Sanity check
	if with_id:
		[(sample, label, __)] = ds.take(1)
	else:
		[(sample, label)] = ds.take(1)

	return ds


ds_train = dataset_from_samples(samples_train).shuffle(shuffle_buffer, seed=shuffle_seed).batch(batch_size).prefetch(5)
ds_valid = dataset_from_samples(samples_valid).batch(1)

# for (sample, label) in ds.take(3):
# 	logger.info(sample, label)

from tensorflow.keras.layers import Conv1D as C, MaxPool1D as M, InputLayer as I

# Predictor model
model = tf.keras.Sequential([
	I(batch_size=None, input_shape=(None, nfeatures)),

	tf.keras.layers.BatchNormalization(),

	M(strides=2, pool_size=5),

	C(strides=1, kernel_size=1, filters=8, kernel_regularizer=tf.keras.regularizers.l2(2)),

	tf.keras.layers.GlobalMaxPooling1D(),

	tf.keras.layers.Dense(len(classes)),

	tf.keras.layers.Softmax(),
])

logger.info(model.summary())

[(sample_batch, label_batch)] = ds_train.take(1)
logger.info(F"sample_batch.shape = {sample_batch.shape}; label_batch.shape = {label_batch.shape}")

optimizer = tf.keras.optimizers.Adam(
	learning_rate=1e-4, amsgrad=True
)

model.compile(
	optimizer=optimizer,
	loss=tf.keras.losses.sparse_categorical_crossentropy,
	metrics=['acc'],
)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=(PARAM['log dir']), histogram_freq=1)
tb_summary_writer = tf.summary.create_file_writer(logdir=str(tb_callback.log_dir))

prediction_history = pd.DataFrame()


def step_callback(epoch, logs):
	prediction = pd.DataFrame(
		data={
			'?': [tuple(predict) for predict in model.predict(ds_valid)],
			'label': [label.numpy().squeeze() for (_, label) in ds_valid.unbatch()],
			'id': samples_valid.index,

			# # Control
			# '?!': [(classes[np.argmax(model.predict(np.asarray([sample.numpy()])).squeeze())], classes[label.numpy().squeeze()]) for (sample, label) in ds_valid.unbatch()],
		}
	).set_index("id")

	prediction['confidence'] = prediction['?'].apply(max)
	prediction['pred_class'] = [classes[c] for c in prediction['?'].apply(np.argmax)]
	prediction['corr_class'] = [classes[c] for c in prediction['label']]
	prediction['is_correct'] = (prediction['?'].apply(np.argmax) == prediction['label'])

	top_misclassified = prediction[~prediction['is_correct']][['confidence', 'corr_class', 'pred_class']]
	top_misclassified = top_misclassified.nlargest(5, columns=['confidence'])
	
	hitheat = prediction.apply(
		lambda row: (
			(0, row['confidence'], 0) if row['is_correct'] else
			(row['confidence'], 0, 0)
		),
		axis=1,
	)

	prediction_history[epoch] = hitheat
	# logger.debug(F"predictions history shape: {prediction_history.shape}")

	image = np.asarray([[list(x) for x in y] for y in prediction_history.T.to_numpy()])
	# logger.debug(F"image shape: {image.shape}")

	with tb_summary_writer.as_default():
		tf.summary.text(
			step=epoch,
			name=F"Top misclassified",
			data=top_misclassified.to_markdown(),
		)
		tf.summary.image(
			step=epoch,
			name=F"Predictions on 'ds_valid'",
			data=image[np.newaxis, ...],
		)


for i in range[1]:
	logger.info(F"Round {i}")

	model.fit(
		ds_train,
		verbose=1, epochs=1000, validation_data=ds_valid,
		callbacks=[tb_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=step_callback)],
	)
