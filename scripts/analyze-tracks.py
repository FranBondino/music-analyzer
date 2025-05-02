import os
import subprocess
import librosa
import essentia.standard as ess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def download_audio(track_name, artist, output_dir="data/audio"):
    """Download audio from YouTube using yt-dlp.
    Returns the path to the downloaded audio file."""
    query = f"{track_name} {artist} track"
    file_name = f"{output_dir}/{query.replace(' ', '_')}.mp3"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_name):
        print(f"Downloading: {query}")
        try:
            subprocess.run([
                "yt-dlp",
                "-x", "--audio-format", "mp3",
                f"ytsearch:{query}",
                "--match-filter", "duration <= 600",  # Limit to 10 minutes
                "-o", file_name
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {query}: {e}")
            return None
    else:
        print(f"Using cached: {query}")
    
    if os.path.exists(file_name):
        print(f"Downloaded audio to: {file_name}")
        return file_name
    else:
        print(f"Audio file not found for {track_name} by {artist}")
        return None

def analyze_audio(file_path):
    """Analyze the audio file to compute features like tempo, energy, danceability, and valence.
    Args:
        file_path (str): Path to the audio file (MP3), either downloaded from YouTube or provided locally.
    """
    try:
        print(f"Analyzing audio file: {file_path}")
        # Load audio with Librosa (for tempo and other features)
        y, sr = librosa.load(file_path, sr=None)

        # Tempo: Librosa (constrain to 110–140 BPM for melodic techno)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        valid_tempos = [t for t in tempos if 110 <= t <= 140]
        final_tempo = round(float(np.mean(valid_tempos))) if valid_tempos else 125
        print(f"Tempo: {final_tempo} BPM")

        # Energy: Librosa (RMS + spectral centroid)
        rms = librosa.feature.rms(y=y)
        energy_normalized = rms.mean() / rms.max()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_max = 10000
        energy_perceptual = 0.7 * energy_normalized + 0.3 * (spectral_centroid / spectral_max)
        energy_perceptual = min(energy_perceptual, 1.0)
        print(f"Energy: {energy_perceptual:.3f}")

        # Danceability: Essentia
        audio = ess.MonoLoader(filename=file_path)()
        danceability_raw, _ = ess.Danceability()(audio)
        print(f"Raw Danceability: {danceability_raw}")
        danceability_normalized = min(danceability_raw / 2.5, 1.0)
        print(f"Danceability: {danceability_normalized:.3f}")

        # Valence Proxy: Essentia (spectral centroid, tempo, energy)
        frame_generator = ess.FrameGenerator(audio, frameSize=2048, hopSize=1024)
        spectrum_algo = ess.Spectrum()
        centroid_algo = ess.Centroid(range=sr/2)
        spectral_centroids = [centroid_algo(spectrum_algo(frame)) for frame in frame_generator]
        centroid_mean = np.mean(spectral_centroids) if spectral_centroids else 3000
        print(f"Spectral Centroid: {centroid_mean:.2f} Hz")

        valence_proxy = (
            0.5 * (centroid_mean / 5000) +  # Brightness
            0.4 * (final_tempo / 140) +     # Tempo
            0.1 * energy_perceptual         # Energy
        )
        valence_proxy = min(max(valence_proxy, 0), 1)
        print(f"Valence Proxy: {valence_proxy:.3f}")

        # Additional features for ML
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=y, sr=sr).mean()

        return {
            "tempo": float(final_tempo),
            "energy": energy_perceptual,
            "danceability": danceability_normalized,
            "valence": valence_proxy,
            "spectral_contrast": spectral_contrast,
            "mfcc": mfcc,
            "file": file_path
        }
    except Exception as e:
        print(f"Analysis failed for {file_path}: {e}")
        return None

# ML: Cluster tracks into groups
def cluster_tracks(df):
    """Cluster tracks into groups (e.g., High-Energy Bangers, Chill Tracks) using K-Means."""
    features = ["tempo", "energy", "danceability", "valence"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_labels = []
    for i, centroid in enumerate(centroids):
        energy, danceability = centroid[1], centroid[2]
        if energy > 0.5 and danceability > 0.5:
            cluster_labels.append("High-Energy Bangers")
        elif energy < 0.4 and danceability < 0.4:
            cluster_labels.append("Chill Tracks")
        else:
            cluster_labels.append("Mid-Tempo Tracks")
    df["cluster_label"] = df["cluster"].map({i: label for i, label in enumerate(cluster_labels)})
    return df

# ML: Classify mood
def classify_mood(df):
    """Classify track mood (e.g., Happy, Sad) using a RandomForest classifier."""
    conditions = [
        (df["valence"] > 0.6) & (df["energy"] > 0.5),
        (df["valence"] < 0.4) & (df["energy"] < 0.4),
        (df["energy"] > 0.6),
        (df["valence"] < 0.5)
    ]
    choices = ["Happy", "Sad", "Energetic", "Calm"]
    df["mood"] = np.select(conditions, choices, default="Neutral")
    features = ["energy", "danceability", "valence", "spectral_contrast", "mfcc"]
    X = df[features]
    y = df["mood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print(f"Mood classifier accuracy: {clf.score(X_test, y_test):.2f}")
    df["mood_predicted"] = clf.predict(X)
    return df

def main(input_csv="data/spotify_tracks.csv", output_csv="data/track_analysis.csv", local_file=None):
    """Analyze tracks from a CSV or a local file, apply ML, and save results."""
    start_time = time.time()

    if local_file:
        # Analyze a single local file
        features = analyze_audio(local_file)
        if features:
            df = pd.DataFrame([{
                "name": "Local Track",
                "artist": "Unknown",
                **features
            }])
            df = cluster_tracks(df)
            df = classify_mood(df)
            print(df[["name", "artist", "tempo", "energy", "danceability", "valence", "cluster_label", "mood_predicted"]])
        return

    # Load tracks from CSV
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df_spotify = pd.read_csv(input_csv)
    tracks = [{"name": row["name"], "artist": row["artist"]} for _, row in df_spotify.iterrows()]
    print(f"Loaded {len(tracks)} tracks from {input_csv}")

    # Analyze tracks one by one
    track_data = []
    for i, track in enumerate(tracks, 1):
        print(f"[{i}/{len(tracks)}] Processing {track['name']} by {track['artist']}")
        
        # Download audio
        audio_file = download_audio(track['name'], track['artist'])
        if not audio_file:
            print(f"Skipping {track['name']} (audio not available)")
            continue
        
        # Analyze audio
        features = analyze_audio(audio_file)
        if features:
            track_data.append({
                "name": track["name"],
                "artist": track["artist"],
                "spotify_id": df_spotify.iloc[i-1]["spotify_id"],
                "popularity": df_spotify.iloc[i-1]["popularity"],
                **features
            })
        
        print(f"Finished analyzing {track['name']} by {track['artist']}")

    # Create DataFrame and apply ML
    if track_data:
        df = pd.DataFrame(track_data)
        
        # Apply ML: Clustering and Mood Classification
        df = cluster_tracks(df)
        df = classify_mood(df)
        
        # Save results to CSV
        df.to_csv(output_csv, index=False)
        print(f"Saved analysis for {len(df)} tracks to {output_csv}")
        
        # Visualize clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df["danceability"], df["energy"], c=df["cluster"], s=100, cmap="viridis")
        plt.xlabel("Danceability")
        plt.ylabel("Energy")
        plt.title("Track Clusters: Energy vs Danceability")
        plt.legend(handles=scatter.legend_elements()[0], labels=df["cluster_label"].unique(), title="Cluster")
        plt.savefig("data/track_clusters_plot.png")
        print("Saved cluster visualization to data/track_clusters_plot.png")
        
        # Print summary statistics
        print(f"Analysis took {time.time() - start_time:.2f} seconds")
        print(df.describe())
    else:
        print("No tracks processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tracks from a CSV or a local audio file, apply ML, and save results.")
    parser.add_argument("--input", type=str, default="data/spotify_tracks.csv", help="Input CSV file path (default: data/spotify_tracks.csv)")
    parser.add_argument("--output", type=str, default="data/track_analysis.csv", help="Output CSV file path (default: data/track_analysis.csv)")
    parser.add_argument("--local_file", type=str, help="Path to a local audio file (e.g., path/to/big_in_japan.mp3)")
    args = parser.parse_args()
    
    main(input_csv=args.input, output_csv=args.output, local_file=args.local_file)