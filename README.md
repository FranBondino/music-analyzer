# Track Analyzer

An open-source tool to analyze track properties (tempo, energy, danceability, valence) using local audio analysis, since Spotify no longer provides these metrics via its API. Retrieve tracks from Spotify playlists or top tracks and generate insights for playlist curation or music analysis.

## Utility
Spotify’s API no longer provides audio features like danceability and energy, limiting tools for music analysis. `track_analyzer` fills this gap by:
- Analyzing tracks locally using `librosa` and `essentia`.
- Helping DJs, curators, and analysts identify bangers (e.g., high danceability tracks for events).
- Providing a free, open-source alternative to paid tools, fostering community contributions.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/track_analyzer.git
   cd track_analyzer