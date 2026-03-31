"""
PHMSA Data Downloader
======================
Run this script LOCALLY on your machine to download the PHMSA pipeline
incident datasets. The cloud environment cannot access phmsa.dot.gov.

Usage:
    python phmsa_data_downloader.py

This will download Gas Transmission and Hazardous Liquid incident data
into the data/raw/ directory.
"""

import os
import requests
import zipfile
import io

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Known PHMSA download URLs (as of 2024-2025)
# These are the direct download links from the PHMSA data portal.
# If URLs change, visit:
#   https://www.phmsa.dot.gov/data-and-statistics/pipeline/
#   distribution-transmission-gathering-lng-and-liquid-accident-and-incident-data

DATASETS = {
    "gas_transmission_2010_present": {
        "url": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/data/incident_gas_transmission_2010_present.zip",
        "description": "Gas Transmission Incidents (2010-Present, Form 7100.2 format)",
    },
    "gas_transmission_1986_2009": {
        "url": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/data/incident_gas_transmission_1986_2009.zip",
        "description": "Gas Transmission Incidents (1986-2009, older format)",
    },
    "hazardous_liquid_2010_present": {
        "url": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/data/incident_hazardous_liquid_2010_present.zip",
        "description": "Hazardous Liquid Incidents (2010-Present)",
    },
    "gas_transmission_flagged": {
        "url": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/flagged/incident_gas_flagged_gt.zip",
        "description": "Gas Transmission Flagged Files (includes trend replication data)",
    },
}


def download_and_extract(name: str, info: dict) -> None:
    """Download a ZIP file and extract to data/raw/."""
    print(f"\nDownloading: {info['description']}")
    print(f"  URL: {info['url']}")

    try:
        resp = requests.get(info["url"], timeout=60)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            extract_dir = os.path.join(DATA_DIR, name)
            os.makedirs(extract_dir, exist_ok=True)
            zf.extractall(extract_dir)
            print(f"  Extracted to: {extract_dir}")
            for f in zf.namelist():
                print(f"    - {f}")

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        print(f"  Try downloading manually from: {info['url']}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHMSA Pipeline Incident Data Downloader")
    print("=" * 60)

    for name, info in DATASETS.items():
        download_and_extract(name, info)

    print("\n" + "=" * 60)
    print("Download complete. Run the EDA notebook next.")
    print("=" * 60)
