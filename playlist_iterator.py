import json
import os
# IMPORTANT from 0 always for build_kg
START = 0
LIMIT = 1000000
def iter_mpd_playlists(mpd_dir: str):
    """Yield playlists from each .json file in mpd_dir (sorted)."""
    files = sorted(
        os.path.join(mpd_dir, f)
        for f in os.listdir(mpd_dir)
        if f.endswith(".json")
    )

    print(f"There is {len(files)} files")
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as fh:
            try:
                data = json.load(fh)
            except Exception as e:
                print(f'Warning: failed to load {fp}: {e}')
                continue
            for pl in data.get("playlists", []):
                if pl["pid"] >= START:
                    yield pl