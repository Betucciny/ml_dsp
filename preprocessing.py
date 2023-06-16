import numpy as np
import librosa
import pandas as pd

genres = ['jazz', 'blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']
columns = ['file_name', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean',
           'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
           'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean',
           'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var',
           'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var',
           'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean',
           'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
           'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
           'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 'label']


def get_data():
    path = './Data/genres_original/'
    num_songs = 100
    df = pd.DataFrame(columns=columns)
    for genre in genres:
        print(f"Processing {genre}...")
        genredf = pd.DataFrame(columns=columns)
        for i in range(0, num_songs):
            song = genre + '.' + str(i).zfill(5) + '.wav'
            if song == 'jazz.00054.wav':
                continue
            songname = path + genre + '/' + song
            data, sample_rate = librosa.load(songname)
            chroma_stft = librosa.feature.chroma_stft(y=data, sr=sample_rate, window='hann', n_fft=2048)
            rms = librosa.feature.rms(y=data)
            spec_cent = librosa.feature.spectral_centroid(y=data, sr=sample_rate, window='hann', n_fft=2048)
            spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate, window='hann', n_fft=2048)
            rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate, window='hann', n_fft=2048)
            zcr = librosa.feature.zero_crossing_rate(data)
            harm = librosa.effects.harmonic(y=data)
            perc = librosa.effects.percussive(y=data)
            mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, window='hann', n_fft=2048)
            tempo, _ = librosa.beat.beat_track(y=data, sr=sample_rate)
            to_append = [f'{song}', np.mean(chroma_stft), np.var(chroma_stft), np.mean(rms), np.var(rms),
                         np.mean(spec_cent), np.var(spec_cent), np.mean(spec_bw), np.var(spec_bw), np.mean(rolloff),
                         np.var(rolloff), np.mean(zcr), np.var(zcr), np.mean(harm), np.var(harm), np.mean(perc),
                         np.var(perc), tempo]
            for e in mfcc:
                to_append.append(np.mean(e))
                to_append.append(np.var(e))
            to_append.append(genre)
            new_row = pd.DataFrame([to_append], columns=columns)
            genredf = pd.concat([genredf, new_row], ignore_index=True, axis=0)
        genredf.to_csv(f'Data/{genre}.csv', index=False)
        df = pd.concat([df, genredf], ignore_index=True, axis=0)
    return df


if __name__ == '__main__':
    get_data()



