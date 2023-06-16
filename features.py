import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from preprocessing import genres


def load_songs():
    songs = []
    for genre in genres:
        y, _ = librosa.load(f'./Data/genres_original/{genre}/{genre}.00000.wav')
        songs.append(y)
    print("Songs loaded", len(songs))
    return songs


def plot_audio(y):
    data = [pd.Series(song) for song in y]

    def plot_func(data, ax, genre):
        ax.plot(data, linewidth=0.5, color='black')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="Audio")


def plot_spectrogram(y):
    def process(song):
        stft = np.abs(librosa.stft(song))
        return librosa.amplitude_to_db(stft, ref=np.max)

    data = [process(song) for song in y]

    def plot_func(data, ax, genre):
        librosa.display.specshow(data, sr=22050, ax=ax, y_axis='log')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="Spectrogram")


def plot_rms(y):
    def process(song):
        rms = librosa.feature.rms(y=song)
        return rms

    data = [process(song) for song in y]

    def plot_func(data, ax, genre):
        ax.plot(data[0], linewidth=0.5, color='black')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="RMS")


def plot_spectral(y):
    def process1(song):
        spectral_centroids = librosa.feature.spectral_centroid(y=song, sr=22050)[0]
        return spectral_centroids

    def process2(song):
        stft = np.abs(librosa.stft(song))
        return librosa.amplitude_to_db(stft, ref=np.max)

    data = [[process1(song), process2(song)] for song in y]

    def plot_func(data, ax, genre):
        centroid = data[0]
        espectral = data[1]
        librosa.display.specshow(espectral, sr=22050, ax=ax, y_axis='log')
        ax.plot(centroid, linewidth=0.5, color='w', label='Spectral Centroid')
        ax.legend(loc='upper right')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="Spectral Centroid")


def plot_bandwidth(y):
    def process(song):

        bandwidth = librosa.feature.spectral_bandwidth(y=song, sr=22050)
        times = librosa.times_like(bandwidth)
        return bandwidth[0], times

    data = [process(song) for song in y]

    def plot_func(data, ax, genre):
        bandwidth = data[0]
        times = data[1]
        ax.semilogy(times, bandwidth, label='Spectral bandwidth')
        ax.set_title(f"Genre: {genre}")
        ax.set(ylabel='Hz')
        ax.legend()

    plot_general(data, plot_func, title="Bandwidth")


def plot_rolloff(y):
    def process(song):
        rolloff = librosa.feature.spectral_rolloff(y=song, sr=22050, roll_percent=0.99)
        rolloff_min = librosa.feature.spectral_rolloff(y=song, sr=22050, roll_percent=0.01)
        return rolloff, rolloff_min

    data = [process(song) for song in y]

    def plot_func(data, ax, genre):
        rolloff, rolloff_min = data
        ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
        ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='b',
                label='Roll-off frequency (0.01)', alpha=0.5)
        ax.set_yscale('log')
        ax.legend(loc='lower right')
        ax.set(title='log Power spectrogram')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="Rolloff")


def plot_mfcc(y):
    def process(song):
        mfccs = librosa.feature.mfcc(y=song, sr=22050, n_mfcc=13)
        return mfccs

    data = [process(song) for song in y]

    def plot_func(data, ax, genre):
        mfccs = data
        librosa.display.specshow(mfccs, sr=22050, ax=ax, x_axis='time')
        ax.set_title(f"Genre: {genre}")

    plot_general(data, plot_func, title="MFCC")


def plot_general(data, plot_func, title=""):
    x, y = [5, 2]
    fig, axes = plt.subplots(x, y, figsize=(20, 20))
    fig.suptitle(title, fontsize=20)
    for i, ax in enumerate(axes.flat):
        plot_func(data[i], ax, genres[i])
    plt.show()


def main():
    songs = load_songs()
    plot_audio(songs)
    plot_spectrogram(songs)
    plot_rms(songs)
    plot_spectral(songs)
    plot_bandwidth(songs)
    plot_rolloff(songs)
    plot_mfcc(songs)


if __name__ == '__main__':
    main()
