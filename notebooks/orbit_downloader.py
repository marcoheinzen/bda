# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
orbit_downloader.py
Downloads precise orbit files (POEORB) from ESA for InSAR processing.
Extracted from Cell 14A2.

Notebook usage:
    from orbit_downloader import run as run_orbit_download
    run_orbit_download(
        orbits_dir=LOCAL_ORBITS_DIR,
        force_rerun=FORCE_RERUN,
    )
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta


def run(orbits_dir, force_rerun=False):
    """
    Args:
        orbits_dir:   Path - LOCAL_ORBITS_DIR
        force_rerun:  bool
    """
    orbits_dir = Path(orbits_dir)
    orbits_dir.mkdir(exist_ok=True, parents=True)
    orbit_log = orbits_dir / "download_log.json"

    if orbit_log.exists() and not force_rerun:
        with open(orbit_log, 'r') as f:
            log = json.load(f)
        print(f"Orbit files already downloaded")
        print(f"  Last updated: {log['timestamp']}")
        print(f"  Total files: {log['total_files']}")
        print(f"  Total size: {log['total_size_gb']:.2f} GB")
        print("\nSet FORCE_RERUN=True to re-download")
    else:
        print("=" * 80)
        print("CELL 14A2: DOWNLOADING SENTINEL-1 ORBIT FILES FROM ESA")
        print("=" * 80)
        print(f"Period: 2021-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Target: {orbits_dir}")

        start_date = datetime(2021, 1, 1)
        end_date = datetime.now()
        s1b_end_date = datetime(2021, 12, 31)  # Sentinel-1B failed Dec 2021

        total_downloaded = 0
        total_skipped = 0

        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month

            # S1A always, S1B only until end of 2021, S1C from Dec 2024
            missions = ['S1A']
            if current_date <= s1b_end_date:
                missions.append('S1B')
            s1c_start_date = datetime(2024, 12, 1)
            if current_date >= s1c_start_date:
                missions.append('S1C')

            for mission in missions:
                mission_dir = orbits_dir / mission / str(year) / f"{month:02d}"
                mission_dir.mkdir(exist_ok=True, parents=True)

                base_url = f"https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/{mission}/{year}/{month:02d}/"

                print(f"\nProcessing {mission} {year}-{month:02d}...")

                month_downloaded = 0
                month_skipped = 0

                try:
                    response = requests.get(base_url, timeout=30)

                    if response.status_code == 404:
                        print(f"  Directory not found (404)")
                        continue

                    if response.status_code != 200:
                        print(f"  HTTP {response.status_code}")
                        continue

                    # Parse HTML directory listing
                    lines = response.text.split('\n')
                    eof_files = []
                    for line in lines:
                        if '.EOF' in line and 'href="' in line:
                            try:
                                filename = line.split('href="')[1].split('"')[0]
                                if filename.endswith('.EOF') or filename.endswith('.EOF.zip'):
                                    eof_files.append(filename)
                            except IndexError:
                                continue

                    if not eof_files:
                        print(f"  No orbit files found")
                        continue

                    for filename in eof_files:
                        orbit_path = mission_dir / filename

                        if orbit_path.exists():
                            month_skipped += 1
                            total_skipped += 1
                            continue

                        orbit_url = base_url + filename
                        orbit_response = requests.get(orbit_url, timeout=60)

                        if orbit_response.status_code == 200:
                            with open(orbit_path, 'wb') as f:
                                f.write(orbit_response.content)
                            month_downloaded += 1
                            total_downloaded += 1
                            print(f"  {filename}")
                        else:
                            print(f"  FAILED: {filename} (HTTP {orbit_response.status_code})")

                        time.sleep(0.1)  # Rate limiting

                    if month_skipped > 0:
                        print(f"  Skipped {month_skipped} existing files")

                except requests.exceptions.Timeout:
                    print(f"  Timeout")
                except requests.exceptions.RequestException as e:
                    print(f"  Network error: {e}")
                except Exception as e:
                    print(f"  Error: {e}")

            current_date = current_date + relativedelta(months=1)

        # Summary
        total_files = len(list(orbits_dir.rglob("*.EOF*")))
        total_size = sum(f.stat().st_size for f in orbits_dir.rglob("*.EOF*"))
        total_size_gb = total_size / (1024**3)

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'total_size_gb': total_size_gb,
            'downloaded': total_downloaded,
            'skipped': total_skipped
        }

        with open(orbit_log, 'w') as f:
            json.dump(log_data, f, indent=2)

        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"  Total files: {total_files}")
        print(f"  Downloaded: {total_downloaded}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Total size: {total_size_gb:.2f} GB")
        print(f"  Location: {orbits_dir}")

    print("Cell 14A2: Orbit download complete")
