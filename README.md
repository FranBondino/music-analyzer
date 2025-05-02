#Music Analyzer Project#


##Overview
The Music Analyzer Project is an open-source Python-based tool designed to analyze audio tracks by extracting musical features and applying machine learning to cluster them for mood-based playlist generation. This project provides a workaround for Spotify's recent restriction on accessing track attributes, allowing users to download audio from YouTube and analyze their own music collections.

##Motivation
With Spotify no longer providing direct access to track attributes for analysis, this project empowers users to extract features independently. It’s ideal for music enthusiasts, DJs, or developers building custom recommendation systems by leveraging audio analysis and clustering.

##Features
Audio Download: Downloads tracks from YouTube using youtube-dl.
Feature Extraction: Extracts musical features (tempo, energy, danceability, valence, spectral centroid) using librosa.
Clustering: Applies k-means clustering to group tracks by energy and danceability.
Statistical Analysis: Summarizes feature distributions (mean, std, min, max, quartiles).
Visualization: Generates scatter plots to visualize clusters using matplotlib.

##Machine Learning Concepts
K-Means Clustering: An unsupervised learning algorithm that groups tracks into clusters based on energy and danceability, enabling mood-based categorization. Users can define the number of clusters based on their dataset.
Feature Selection: Energy and danceability are used as primary features, as they influence a track’s mood and dance potential, customizable with additional features if desired.

##Statistical Concepts
Descriptive Statistics: Calculates mean, standard deviation, minimum, maximum, and quartiles for all extracted features, providing insights into the dataset’s distribution.
Quartile Analysis: Helps identify data spread and outliers, allowing users to refine their analysis.

##Data Analysis Concepts
Feature Extraction: Utilizes librosa to compute audio features:
Tempo: Mean beats per minute (BPM).
Energy: Average root mean square (RMS) energy, representing intensity.
Danceability: Derived by normalizing tempo relative to 120 BPM, scaled between 0 and 1.
Spectral Centroid: Mean frequency weighted by amplitude, indicating brightness.
Valence Proxy: Mean spectral flatness, approximating emotional tone.
Data Visualization: Creates scatter plots (energy vs. danceability) to interpret clustering results.
Pipeline Automation: Automates downloading, analyzing, and saving results to CSV for reproducibility.

##Installation
Clone the repository:git clone https://github.com/yourusername/music-analyzer.git
cd music-analyzer

Install dependencies:pip install -r requirements.txt
Note: Ensure ffmpeg is installed for audio conversion:sudo apt-get install ffmpeg

Prepare your own CSV file with track names, artists, and optional Spotify IDs (e.g., tracks.csv) and place it in the data/ directory. Use the format: name,artist,spotify_id (Spotify ID is optional).

##Usage
Run the analysis script with your custom input:
python3 scripts/analyze-tracks.py --input data/tracks.csv --output data/track-analysis.csv

Input: Path to your CSV file with track information.
Output: Path to save the analysis results (CSV format).
Output Files:
data/track-analysis.csv: Analyzed features and cluster labels.
data/track_clusters_plot.png: Scatter plot of clusters.

##Project Structure
data/: Directory for input CSV and output files (users to populate).
scripts/: Contains the main script (analyze-tracks.py).
requirements.txt: Lists Python dependencies.

##Getting Started
Create a tracks.csv file with your desired tracks (e.g., name,artist,spotify_id).
Run the script to download and analyze your tracks, then explore the output CSV and plot.
Customize the clustering (e.g., number of clusters) by modifying the script if needed.

##Significance
This project is valuable because Spotify restricts track attribute access, offering an independent solution for audio analysis. Users can analyze any track collection, making it a versatile tool for personal or research purposes.

##Future Improvements
Add support for additional features (e.g., timbre, key).
Enhance danceability calculation with rhythm analysis.
Support batch processing for larger datasets.

##Dependencies
Python 3.8+
Libraries: librosa, youtube-dl, pandas, numpy, matplotlib, scikit-learn
