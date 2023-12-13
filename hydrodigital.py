import os
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from pydub import AudioSegment
from keras import layers, models
from pathlib import Path

# Change those settings to match your own directory
audio_dir = '/mnt/c/Users/micha/Downloads/normalized/'
chopped_dir = '/mnt/c/Users/micha/Downloads/chopped/'


# this function resamples audio from a original sample rate to a target sample rate
def resample_audio(wav, orig_sr, target_sr):
    resampled_wav = signal.resample(wav, int(len(wav) * target_sr / orig_sr))
    return resampled_wav


# returns a TensorFlow Tensor which contains the audio waveform after it
# has been resampled to a mono signal with a sampling rate of 16000 Hz
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav_np = tf.numpy_function(np.squeeze, [wav], tf.float32)
    # Resample to 16000Hz
    resampled_wav = tf.numpy_function(resample_audio, [wav_np, sample_rate, 16000], tf.float32)
    # Convert back to TensorFlow tensor
    resampled_wav = tf.convert_to_tensor(resampled_wav)
    return resampled_wav


def split_and_save_audio(input_audio_file, category, num_segments):
    Path(chopped_dir+category).mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(input_audio_file)

    for i in range(num_segments):
        t1 = i * 3000  # Start time in milliseconds
        t2 = (i + 1) * 3000  # End time in milliseconds
        segment = audio[t1:t2]

        output_file = os.path.join(chopped_dir + category, f'{category}_{i}chopped.wav')
        segment.export(output_file, format="wav")


# scans the audio directory and splits each file into 3 second segments
def scan_splice():
    files = []
    for f in os.listdir(audio_dir):
        files.append(f)
    num_segments = 100  # length of audio divided by 3 seconds
    for i in files:
        split_and_save_audio(audio_dir + i, i.split('.')[0], num_segments)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def load_data():
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        chopped_dir,
        batch_size=5,
        validation_split=0.2,
        seed=0,
        output_sequence_length=44100,
        subset='both',
        shuffle=True
    )
    class_names = np.array(train_ds.class_names)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    return train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds, class_names, train_ds, val_ds, test_ds

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


def visualize_spectrogram(ds, class_names):
    example_audio,example_label = next(iter(train_ds))
    waveform = example_audio[0]
    label = class_names[example_label[0]]
    spectrogram = get_spectrogram(waveform)
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.suptitle(label.title())
    plt.show()


def train(train_spectrogram_ds, val_spectrogram_ds, class_names, epochs):
    input_shape = list(train_spectrogram_ds.take(1))[0][0].shape[1:]
    num_labels = len(class_names)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=epochs,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    return model, history


if __name__ == '__main__':
    epochs = 5
    # Run Scan Splice on first run to splice the data
    scan_splice()
    # train_spectrogram_ds, val_spectrogram_ds, test_spectrogram_ds, class_names, train_ds, val_ds, test_ds = load_data()

    # To Visualize Spectrogram
    # visualize_spectrogram(val_ds, class_names)

    # Model Training and Evaluation
    # model, history = train(train_spectrogram_ds, val_spectrogram_ds, class_names, epochs)
    # model.evaluate(test_spectrogram_ds, return_dict=True)
