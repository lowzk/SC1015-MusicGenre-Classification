# SC1015 Music Genre Classification
Mini-Project for the SC1015 Module on Data Science and Artificial Intelligence. This project was done by Low Zhe Kai ([@lowzk](https://github.com/lowzk)) and Marc Chern ([@Trigon25](https://github.com/Trigon25)).

Video Presentation: [link](https://www.youtube.com/watch?v=1p3GvRDHhvE&ab_channel=LowZheKai)

Dataset obtained from Kaggle: [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---
## Problem Formulation

For a long time, experts have been trying to quantify the difference in sounds and what differentiates one genres from another. But it has been hard to do so due to the **subjectivity** and **complexity** of music…

As such, we aimed to accomplish **three** objectives:

1. Undergo an **in-depth analysis** of the sounds of different genres 
2. **Visualize** and **quantify** their differences by extracting relevant features
3. Build a machine learning model that can **classify** the genre of an audio file

---
## Approach Taken
1. Data Collection
2. Raw Data Extraction and Preparation
3. Signal Processing
4. Audio Feature Extraction
5. Exploratory Data Analysis
6. Model Testing
---
### 1. Data Collection
We collected our data from Kaggle, using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) obtained from Kaggle.

This dataset is the most used public dataset used for music genre classification.  
<br />

### 2. Raw Data Extraction and Preparation
We used the **Librosa** python library, and took the audio files and sampled them. This gives the audio file as an array of floats. Thereafter, this array can be used in further feature extractions.  

During our data preparation, we realised that one of the audio files, `jazz0054.wav`, was corrupted. Therefore, we removed that audio file.  
<br />

### 3. Signal Processing
Fourier transformation is a mathematical transformation which splits an audio signal into it's component frequencies. However, in music, the frequencies are everchanging.  

Therefore, we have to use a **Short-Time Fourier transformation (STFT)**. This uses a moving window and calculates the frequecy components in that window.  
<br />

### 4. Audio Feature Extraction
We extracted many different features from the audio files. These included:

1. **Root-Mean-Squared** and **Variance** of raw audio
2. **Zero Crossing Rate**
3. **Tempo (BPM)**
4. **Harmonics and Percussion Source Separation**
5. **Chroma STFT**
6. **Spectral Centroid**
7. **Spectral Rolloff**
8. **Mel-Frequency Cepstral Coefficients**  

We attempted to do both `PCA` and `Feature selection` using a Selector Function to reduce the dimensionality of our features. However, after futher testing, we realised that using all our extracted features gave us better performance during our accuracy testing.

All of the above feature extractions and attempted dimensionality reductions can be found in our notebook.  
<br />

### 5. Exploratory Data Analysis (EDA)
In our EDA, we plotted `boxplots` for the different features against the different genres. This allowed us to identify the outliers in our dataset. However, as our dataset is really small, and due to the nature of music being subjective, we realised that it would be `incorrect` to remove the outliers in our specific case. Therefore, our boxplots allowed us to see how different features were distributed differently across different genres.

We also plotted the `correlation matrix` with of some of our features extracted, to see the different correlations between our extracted features. For example, the RMS value had a strong positive correlation with Harmonics and Percussion Source Separation values, which makes sense as the separation would mostly keep the same shape of the audio signals, but just splittting it.

Both these plots can be seen in our notebook.  
<br />

### 6. Model Testing
For model testing, we compared the use of many different models and their accuracies. This allowed us to choose the best model for our application.

The models tested are:
1. Naive Bayes
2. Stochastic Gradient Descent
3. K Nearest Neighbours
4. Decision Trees
5. Random Forest Regression
6. Support Vector Machine
7. Logistic Regression
8. XGBoost (Extreme Gradient Boosting)
9. XGBoost using Random Forest

The best model was ultimately model 8, XGBoost. Our results can be seen in the notebook.  
<br />

---
## Conclusions
We tested our model with a sample audio file, `reggae01.wav`. It managed to predict the genre correctly.

Future uses of the model include:
1. Online music streaming services can automatically classify music genres, eliminating the need for manual input of metadata information
2. Research insights into the features which characterize genres
3. The development of AI Music generation software, using better understanding of music genres and features in music

---
## References
1. https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. https://www.nvidia.com/en-us/glossary/data-science/xgboost/
3. https://www.researchgate.net/profile/Derry-Fitzgerald/publication/254583990_HarmonicPercussive_Separation_using_Median_Filtering/links/00b495396ef03235af000000/Harmonic-Percussive-Separation-using-Median-Filtering.pdf
4. http://librosa.org/doc/0.8.1/index.html
5. https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
