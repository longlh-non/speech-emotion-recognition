# -*- coding: utf-8 -*-
"""speech-emotion-recognition.ipynb
Original file is located on Kaggle
"""

import subprocess
import sys

packages = [("numpy", ""), ("matplotlib", "3.7"), ("keras", ""), ("opencv-python", ""), ("kaggle", ""), ("soundfile", ""), ("librosa", ""), ("pandas", ""),
            ("seaborn", ""), ("IPython", ""), ("sounddevice", ""), ("pydub", "")]


def install(package, version=None):
    if version is not None and version != "":
        package += f"=={version}"
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for p, p_version in packages:
    install(p, p_version)

"""# Importing Libraries"""

import os
import sys

import pandas as pd
import numpy as np
from zipfile import ZipFile
import seaborn as sns
import matplotlib.pyplot as plt
import kaggle
import time

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio
from pydub import AudioSegment
import tkinter as tk
from tkinter import Button
import sounddevice as sd

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""## Connect to Kaggle and download dataset (local only)
"""


def connect_to_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    return api


def download_dataset_from_kaggle(kaggle_api, dataset_path):
    # downloading datasets
    print(f"Downloading dataset from {dataset_path}")
    kaggle_api.dataset_download_files(dataset_path)


def unzip_dataset(dataset_path):
    dataset_path_split = dataset_path.split('/')
    zf = ZipFile(f'{dataset_path_split[1]}.zip')
    print(f"Extract dataset from zip file: {dataset_path_split[1]}")
    zf.extractall(f'dataset\\{dataset_path_split[1]}\\')  # save files in selected folder
    zf.close()


datasets = ['ejlok1/surrey-audiovisual-expressed-emotion-savee',
            'ejlok1/cremad',
            'ejlok1/toronto-emotional-speech-set-tess',
            'uwrfkaggler/ravdess-emotional-speech-audio']

api = connect_to_kaggle()
for dataset in datasets:
    download_dataset_from_kaggle(api, dataset)
    unzip_dataset(dataset)

"""## Data Preparation
"""

# Paths for data.
Ravdess = "dataset/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "dataset/cremad/AudioWAV/"
Tess = ("dataset/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional "
        "speech set data/")
Savee = "dataset/surrey-audiovisual-expressed-emotion-savee/ALL/"

"""## 1. Ravdess Dataframe
"""

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace(
    {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
Ravdess_df.head()

"""## 2. Crema DataFrame"""

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()

"""## 3. TESS dataset"""

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()

"""## 4. CREMA-D dataset
"""

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()

# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)
data_path.to_csv("data_path.csv", index=False)
data_path.head()

"""## Data Visualisation and Exploration
"""

plt.title('Count of Emotions', size=16)
sns.countplot(data=data_path, x="Emotions")
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()


def create_waveshow(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('waveshow for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)


def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    x_scope = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x_scope))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


def visualize_data(data, sr, e):
    create_spectrogram(data, sr, e)
    create_waveshow(data, sr, e)


def show_audio_window(x, sample_rate, audio_type):
    def play_audio():
        # Play the audio
        sd.play(x, sample_rate)
        sd.wait()
        window.destroy()

    # Create the main window
    print(f"Please check tkinter window for {audio_type} audio playing")
    window = tk.Tk()
    window.title(f"{audio_type} audio")

    # Create a label with some initial text
    label = tk.Label(window, text=f"{audio_type} audio", font=("Helvetica", 12))
    label.pack(pady=10)
    # Create a button to start audio playback
    play_button = Button(window, text="Play Audio", command=play_audio)
    play_button.pack(pady=10)

    # Run the GUI
    window.mainloop()


emotion = 'fear'
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
visualize_data(data, sampling_rate, emotion)
show_audio_window(data, sampling_rate, emotion)

emotion = 'angry'
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
visualize_data(data, sampling_rate, emotion)
show_audio_window(data, sampling_rate, emotion)

emotion = 'sad'
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
visualize_data(data, sampling_rate, emotion)
show_audio_window(data, sampling_rate, emotion)

emotion = 'happy'
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
visualize_data(data, sampling_rate, emotion)
show_audio_window(data, sampling_rate, emotion)

plt.show()


# Data Augmentation
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)

# 1. Original Audio

plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=data, sr=sample_rate)
plt.title("Original Audio")
show_audio_window(data, sample_rate, "Original")

# 2. Noise Injection

x = noise(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
plt.title("Noise Injection")
show_audio_window(x, sample_rate, "Noise Injection")

# 3. Stretching
x = stretch(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
plt.title("Stretching")
show_audio_window(x, sample_rate, "Stretching")

# 4. Shifting
x = shift(data)
plt.figure(figsize=(14, 4))
plt.title("Shifting")
librosa.display.waveshow(y=x, sr=sample_rate)
show_audio_window(x, sample_rate, "Shifting")

# 5. Pitching
x = pitch(data, sample_rate)
plt.figure(figsize=(14, 4))
plt.title("Pitching")
librosa.display.waveshow(y=x, sr=sample_rate)
show_audio_window(x, sample_rate, "Pitching")

plt.show()


def time_converter(time_value):
  return f"{time_value: .4f} seconds" if time_value < 1.0 else time.strftime("%H:%M:%S", time.gmtime(time_value))


"""
## Feature Extraction
"""


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

start_time = time.time()
print(f"Extracting features would take some times to finish, please wait. Start at: {time_converter(start_time)}")
X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

len(X), len(Y), data_path.Path.shape

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
Features.head()

extracting_time = time.time() - start_time
print(f"Finish feature extracting after {time_converter(extracting_time)}, please check features.csv file if needed.")

"""
## Data Preparation
"""

start_time = time.time()
print(f"Data preparation would take some times to finish, please wait. Start at: {time_converter(start_time)}")
X = Features.iloc[:, :-1].values
Y = Features['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

preparing_time = time.time() - start_time
print(f"Data preparation finished after {time_converter(preparing_time)}.")

"""## Modelling"""
model = Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=8, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)


start_time = time.time()
print(f"Training model, please wait. Start at: {time_converter(start_time)}")
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])
training_time = time.time() - start_time
print(f"Training finish after: {time_converter(training_time)}")
print("Accuracy of our model on test data : ", model.evaluate(x_test, y_test)[1] * 100, "%")

epochs = [i for i in range(50)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20, 6)
ax[0].plot(epochs, train_loss, label='Training Loss')
ax[0].plot(epochs, test_loss, label='Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs, train_acc, label='Training Accuracy')
ax[1].plot(epochs, test_acc, label='Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test, y_pred))
