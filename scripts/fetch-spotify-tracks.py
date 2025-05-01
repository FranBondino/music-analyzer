import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Validate credentials
if not client_id or not client_secret:
    raise ValueError("Missing Spotify API credentials in .env")

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def get_playlist_tracks(query, limit=50):
    """Search for a Spotify playlist by query and fetch its tracks."""
    try:
        results = sp.search(q=query, type="playlist", limit=1)
        if not results["playlists"]["items"]:
            raise ValueError(f"No playlist found for query: {query}")
        
        playlist_id = results["playlists"]["items"][0]["id"]
        playlist_name = results["playlists"]["items"][0]["name"]
        print(f"Fetching tracks from playlist: {playlist_name}")

        # Get tracks
        tracks = []
        offset = 0
        while True:
            results = sp.playlist_tracks(playlist_id, offset=offset, limit=100)
            for item in results["items"]:
                track = item["track"]
                if track:
                    tracks.append({
                        "name": track["name"],
                        "artist": track["artists"][0]["name"],
                        "spotify_id": track["id"],
                        "popularity": track["popularity"]
                    })
            offset += 100
            if len(results["items"]) < 100 or len(tracks) >= limit:
                break
        
        return tracks[:limit]
    except Exception as e:
        print(f"Failed to fetch tracks for query {query}: {e}")
        return []

def main(query, output_csv="data/spotify_tracks.csv"):
    """Fetch tracks from a Spotify playlist and save to a CSV."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Fetch tracks
    print(f"Searching for playlist with query: {query}")
    tracks = get_playlist_tracks(query, limit=50)
    if not tracks:
        print("No tracks retrieved. Exiting.")
        return
    
    # Save to CSV
    df = pd.DataFrame(tracks)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(tracks)} tracks to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch tracks from a Spotify playlist and save to CSV.")
    parser.add_argument("--query", type=str, required=True, help="Search query for Spotify playlist (e.g., 'Top 50 Melodic Techno')")
    parser.add_argument("--output", type=str, default="data/spotify_tracks.csv", help="Output CSV file path (default: data/spotify_tracks.csv)")
    args = parser.parse_args()
    
    main(query=args.query, output_csv=args.output)