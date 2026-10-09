"""Fetch the 'Network operator KPIs time series' dataset (Zenodo 8147768).

CC BY-NC 4.0. 6 MB zip, downloads in <1 min. Staged under
experiments/final2/data/kpiops/. Kept out of git (third-party data).
"""

import os
import io
import sys
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data", "kpiops")
URL = ("https://zenodo.org/records/8147768/files/"
       "network_operator_KPIs_time_series_dataset.zip?download=1")


def main():
    os.makedirs(DEST, exist_ok=True)
    if os.path.exists(os.path.join(DEST, "data_real_incidents.txt")):
        print("already present:", DEST)
        return
    print("downloading", URL, flush=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            rel = name.split("/", 1)[1] if "/" in name else name
            if not rel:
                continue
            out = os.path.join(DEST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
    print("done ->", DEST)


if __name__ == "__main__":
    main()
    sys.exit(0)
