import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from PIL import Image
import pathlib
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import keras
import warnings
#Initializng the feature names in the csv file
header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
#Create the csv file and write the header
file = open('test_data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
#The different kind of genres to process
genres = 'Prog Non-Prog'.split()
for g in genres:
    for filename in os.listdir(f'C:/Users/Surya/Desktop/ML_Dataset/testset2/{g}/'):#Enter the genre directory
        songname = f'C:/Users/Surya/Desktop/ML_Dataset/testset2/{g}/{filename}'#Fetch each file
        y, sr = librosa.load(songname, mono=True)#Load each song, can set duration to be loaded using 'duration' parameter 
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        #Labelling Prog as 1 and NonProg as 0
        if(g == "Prog"):
            to_append += f' 1'
        else:
            to_append += f' 0'
        #Write all the computed features to the csv file
        file = open('test_data.csv', 'a', newline='',encoding="utf-8")
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())