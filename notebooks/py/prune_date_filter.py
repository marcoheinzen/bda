# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# prune_date_filter.py
# Shared date filtering for prune notebooks (NB03d, NB03f, NB05a2).
# Loads battle dates from AOI.geojson, computes valid temporal window,
# checks if TIF filenames have dates inside/outside the window.
# Works for ALL cities (bonus and non-bonus).
#
# Usage:
#   from bonus_city_filter import DateFilter
#   df = DateFilter(CITIES_DIR, buffer_pre=90, buffer_post=90)
#   win = df.get_window('Mariupol')
#   outside, reason = df.check_file('s1__coh__20220115_20220127.tif', 'Mariupol')

import re
from datetime import datetime, timedelta
from pathlib import Path

RE_DATE8 = re.compile(r'(\d{8})')


class DateFilter:
    def __init__(self, cities_dir, buffer_pre=90, buffer_post=90):
        self.cities_dir = Path(cities_dir)
        self.buffer_pre = buffer_pre
        self.buffer_post = buffer_post
        self._cache = {}

    def _load_battle_dates(self, city_name):
        aoi_path = self.cities_dir / city_name / 'AOI.geojson'
        if not aoi_path.exists():
            return None, None, None
        with open(str(aoi_path), 'r') as f:
            raw = f.read(50000)
        bs = None
        be = None
        tier = None
        m = re.search(r'"battle_start"\s*:\s*"([^"]+)"', raw)
        if m:
            try:
                bs = datetime.strptime(m.group(1)[:10], '%Y-%m-%d')
            except ValueError:
                pass
        m = re.search(r'"battle_stop"\s*:\s*"([^"]+)"', raw)
        if m:
            val = m.group(1)
            if val != 'ongoing':
                try:
                    be = datetime.strptime(val[:10], '%Y-%m-%d')
                except ValueError:
                    pass
        m = re.search(r'"tier"\s*:\s*(\d+)', raw)
        if m:
            tier = int(m.group(1))
        return bs, be, tier

    def get_battle_dates(self, city_name):
        """Returns (battle_start, battle_stop, tier). Cached."""
        if city_name not in self._cache:
            self._cache[city_name] = self._load_battle_dates(city_name)
        return self._cache[city_name]

    def get_window(self, city_name):
        """Returns (win_start, win_end) datetime tuple. None if no bound."""
        bs, be, _ = self.get_battle_dates(city_name)
        win_start = (bs - timedelta(days=self.buffer_pre)) if bs else None
        win_end = (be + timedelta(days=self.buffer_post)) if be else None
        return win_start, win_end

    @staticmethod
    def extract_dates(fname):
        """Extract all YYYYMMDD dates from a filename. Returns list of datetime."""
        dates = []
        for m in RE_DATE8.finditer(fname):
            dstr = m.group(1)
            try:
                dt = datetime.strptime(dstr, '%Y%m%d')
                if 2015 <= dt.year <= 2030:
                    dates.append(dt)
            except ValueError:
                pass
        return dates

    def check_file(self, fname, city_name):
        """Check if a TIF file has dates outside the valid window.

        For COH pairs (2 dates in filename): prune only if ALL dates are outside.
        A cross-battle pair with one date in-window is valid and kept.

        Returns:
            (is_outside, reason_str) tuple.
            is_outside=True means file should be pruned.
        """
        dates = self.extract_dates(fname)
        if not dates:
            return False, ''
        win_start, win_end = self.get_window(city_name)
        all_outside = True
        reason_parts = []
        for dt in dates:
            dt_outside = False
            if win_start and dt < win_start:
                dt_outside = True
                reason_parts.append(
                    f'{dt.strftime("%Y-%m-%d")} before {win_start.strftime("%Y-%m-%d")}'
                )
            elif win_end and dt > win_end:
                dt_outside = True
                reason_parts.append(
                    f'{dt.strftime("%Y-%m-%d")} after {win_end.strftime("%Y-%m-%d")}'
                )
            if not dt_outside:
                all_outside = False
        return all_outside, '; '.join(reason_parts)

    def classify_sensor(self, fname):
        """Classify TIF by sensor group from __ naming convention."""
        fl = fname.lower()
        if fl.startswith('s1__coh'):
            return 'COH'
        if fl.startswith('s1__card') or fl.startswith('s1__vv') or fl.startswith('s1__vh'):
            return 'CARD'
        if fl.startswith('s2__'):
            return 'MS'
        if fl.startswith('landuse__') or fl.startswith('lulc__'):
            return 'LANDUSE'
        if fl.startswith('bldg__') or fl.startswith('building'):
            return 'BUILDING'
        if fl.startswith('damage') or fl.startswith('unosat'):
            return 'LABEL'
        if 'coh' in fl:
            return 'COH'
        if 'card' in fl or 'backscatter' in fl:
            return 'CARD'
        if any(b in fl for b in ['_b02', '_b03', '_b04', '_b05', '_b06', '_b07',
                                  '_b08', '_b8a', '_b11', '_b12', '_scl']):
            return 'MS'
        if 'ndvi' in fl or 'nbr' in fl or 'bsi' in fl or 'ndwi' in fl:
            return 'INDEX'
        if 'composite' in fl or 'rgb' in fl:
            return 'COMPOSITE'
        if 'rolling' in fl or 'zscore' in fl or 'baseline' in fl:
            return 'TEMPORAL'
        return 'OTHER'

    def scan_dir(self, base_dir, city_filter=None):
        """Scan base_dir/{city}/**/*.tif. Returns (audit_rows, prune_candidates).

        audit_rows: list of dicts with per-city stats.
        prune_candidates: list of (city, tif_path, dates, reason) for outside files.
        """
        base_dir = Path(base_dir)
        audit_rows = []
        prune_candidates = []

        city_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()
                            and d.name not in ('metadata', 'desktop.ini',
                                                'dataset', '.ipynb_checkpoints')])
        if city_filter:
            city_filter_set = set(city_filter)
            city_dirs = [d for d in city_dirs if d.name in city_filter_set]

        for city_dir in city_dirs:
            city = city_dir.name
            bs, be, tier = self.get_battle_dates(city)
            win_start, win_end = self.get_window(city)

            from collections import defaultdict
            sensor_counts = defaultdict(int)
            dated_count = 0
            undated_count = 0
            outside_count = 0
            total_bytes = 0
            outside_bytes = 0
            date_range_min = None
            date_range_max = None

            tifs = list(city_dir.rglob('*.tif'))

            for tif in tifs:
                fname = tif.name
                sensor = self.classify_sensor(fname)
                sensor_counts[sensor] += 1
                fsize = tif.stat().st_size
                total_bytes += fsize

                dates = self.extract_dates(fname)
                if not dates:
                    undated_count += 1
                    continue

                dated_count += 1

                for dt in dates:
                    if date_range_min is None or dt < date_range_min:
                        date_range_min = dt
                    if date_range_max is None or dt > date_range_max:
                        date_range_max = dt

                file_outside, reason = self.check_file(fname, city)
                if file_outside:
                    outside_count += 1
                    outside_bytes += fsize
                    prune_candidates.append((city, tif, dates, reason))

            row = {
                'city': city,
                'tier': tier,
                'battle_start': bs.strftime('%Y-%m-%d') if bs else None,
                'battle_stop': be.strftime('%Y-%m-%d') if be else 'ongoing',
                'total_tifs': len(tifs),
                'total_mb': total_bytes / (1024 * 1024),
                'dated': dated_count,
                'undated': undated_count,
                'outside': outside_count,
                'outside_mb': outside_bytes / (1024 * 1024),
                'date_min': date_range_min.strftime('%Y-%m-%d') if date_range_min else None,
                'date_max': date_range_max.strftime('%Y-%m-%d') if date_range_max else None,
                'sensors': dict(sensor_counts),
            }
            audit_rows.append(row)

        return audit_rows, prune_candidates
