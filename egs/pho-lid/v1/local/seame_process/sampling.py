import librosa
import soundfile
import os

def resample(audio):
    data, sr = librosa.load(audio, sr=None)
    # new_name = './local/seame_process/text_audio.wav'
    soundfile.write(audio, data, 16000, subtype='PCM_16')


if __name__ == "__main__":
    data_root = './data/seame/'
    for folder in os.listdir(data_root):
        folder_path = data_root + folder
        pure_folder = folder_path + '/pure/'
        cat_folder = folder_path + '/cat/'

        for filename in os.listdir(pure_folder):
            if filename.endswith('.wav'):
                resample(pure_folder + filename)
            elif filename.endswith('.npy'):
                os.remove(pure_folder + filename)
        
        for filename in os.listdir(cat_folder):
            if filename.endswith('.wav'):
                resample(cat_folder + filename)
            elif filename.endswith('.npy'):
                os.remove(cat_folder + filename)

        print(f'Finish resampling for {folder}')

print('SEAME Resampling done.')