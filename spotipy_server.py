from flask import Flask, request
import webbrowser
from spotipy.oauth2 import SpotifyOAuth
import spotipy

# Flask app to catch the redirect
app = Flask(__name__)
token_info = None

with open("api_keys", "r") as f:
    lines = [line.strip() for line in f.readlines()]

CLIENT_ID = lines[0]
CLIENT_SECRET = lines[1]
REDIRECT_URI = lines[2]

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read playlist-modify-public"
)

@app.route("/")
def callback():
    global token_info
    code = request.args.get("code")
    token_info = sp_oauth.get_access_token(code)
    return "Authorization successful! You can close this tab."


if __name__ == "__main__":
    # Open browser for Spotify login
    auth_url = sp_oauth.get_authorize_url()
    webbrowser.open(auth_url)

    # Run Flask server to catch the redirect
    app.run(port=8888, debug=False)
