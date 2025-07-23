from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

class Cell:
    ALL_DIRECTIONS = [
        (1, 0),   # E
        (1, 1),   # NE
        (0, 1),   # N
        (-1, 1),  # NW
        (-1, 0),  # W
        (-1, -1), # SW
        (0, -1),  # S
        (1, -1),  # SE
    ]

    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

        # directions grouped by time categories
        # Each maps time key → Counter of direction vectors (dx, dy)
        self.directions_by_time = {
            'hourly': defaultdict(Counter),       # 0–23
            'time_block': defaultdict(Counter),   # '00-08', '08-12', '12-16, '16-20', '20-24'
            'weekday': defaultdict(Counter),      # 0–6 (Mon–Sun)
            'daily': defaultdict(Counter),        # datetime.date()
        }

        self.velocities_by_time = {
            'hourly': defaultdict(list),
            'time_block': defaultdict(list),
            'weekday': defaultdict(list),
            'daily': defaultdict(list),
        }

        self.volumes_by_time = {
            'hourly': defaultdict(int),
            'time_block': defaultdict(int),
            'weekday': defaultdict(int),
            'daily': defaultdict(int),
            }
        
        self.timegroup_key_counts = {
            'hourly': defaultdict(int),
            'time_block': defaultdict(int),
            'weekday': defaultdict(int),
            'daily': defaultdict(int),
        }
        
        self.seen_time_keys = {
            'hourly': set(),       # stores (hour, date)
            'time_block': set(),   # stores (block, date)
            'weekday': set(),      # stores (weekday, date)
            'daily': set(),        # stores (date,)
        }

    def update_direction(self, timestamp: datetime, dx: int, dy: int):
        """Update directions for all relevant time groups."""
        hour = timestamp.hour
        block = self.get_time_block(timestamp)
        weekday = timestamp.weekday()
        date = timestamp.date()

        self.directions_by_time['hourly'][hour][(dx, dy)] += 1
        self.directions_by_time['time_block'][block][(dx, dy)] += 1
        self.directions_by_time['weekday'][weekday][(dx, dy)] += 1
        self.directions_by_time['daily'][date][(dx, dy)] += 1

    def update_velocity(self, timestamp: datetime, speed: float):
        hour = timestamp.hour
        block = self.get_time_block(timestamp)
        weekday = timestamp.weekday()
        date = timestamp.date()

        self.velocities_by_time['hourly'][hour].append(speed)
        self.velocities_by_time['time_block'][block].append(speed)
        self.velocities_by_time['weekday'][weekday].append(speed)
        self.velocities_by_time['daily'][date].append(speed)

    def update_volume(self, timestamp: datetime):
        hour = timestamp.hour
        block = self.get_time_block(timestamp)
        weekday = timestamp.weekday()
        date = timestamp.date()

        self.volumes_by_time['hourly'][hour] += 1
        self.volumes_by_time['time_block'][block] += 1
        self.volumes_by_time['weekday'][weekday] += 1
        self.volumes_by_time['daily'][date] += 1

        # Count time key occurrences only once per (key, date)
        key_date_pairs = {
            'hourly': (hour, date),
            'time_block': (block, date),
            'weekday': (weekday, date),
            'daily': (date,),  # tuple for consistency
        }

        for group, pair in key_date_pairs.items():
            if pair not in self.seen_time_keys[group]:
                self.seen_time_keys[group].add(pair)
                self.timegroup_key_counts[group][pair[0]] += 1

    def get_dominant_direction(self, time_group: str, key) -> tuple:
        """Return the (dx, dy) direction with the highest count."""
        directions = self.directions_by_time.get(time_group, {}).get(key)
        if not directions:
            return None
        return max(directions.items(), key=lambda x: x[1])[0]

    def get_direction_distribution(self, time_group: str, key) -> dict:
        """Return normalized transition probabilities."""
        counter = self.directions_by_time.get(time_group, {}).get(key)
        if not counter or sum(counter.values()) == 0:
            # Return zeroed dict with all directions
            return {d: 0.0 for d in Cell.ALL_DIRECTIONS}

        total = sum(counter.values())
        return {
            d: counter.get(d, 0) / total
            for d in Cell.ALL_DIRECTIONS
        }
    
    def get_velocity_histogram(self, time_group: str, key, bins=None, range=(0, 3.0)) -> np.ndarray:
        values = self.velocities_by_time.get(time_group, {}).get(key, [])
        if not values:
            return np.zeros(bins, dtype=np.float64)

        hist, _ = np.histogram(values, bins=bins, range=range, density=True)
        total = hist.sum()
        if total == 0:
            return np.zeros_like(hist, dtype=np.float64)
        return hist
    
    def get_volume(self, time_group: str, key) -> int:
        return self.volumes_by_time.get(time_group, {}).get(key, 0)
    
    def get_timegroup_key_count(self, time_group: str, key) -> int:
        return self.timegroup_key_counts.get(time_group, {}).get(key, 0)
    
    def update_with(self, other_cell):
        for group in other_cell.directions_by_time:
            for key, counter in other_cell.directions_by_time[group].items():
                self.directions_by_time[group][key].update(counter)

        for group in other_cell.velocities_by_time:
            for key, values in other_cell.velocities_by_time[group].items():
                self.velocities_by_time[group][key].extend(values)

        for group in other_cell.volumes_by_time:
            for key, volume in other_cell.volumes_by_time[group].items():
                self.volumes_by_time[group][key] += volume
        
        for group in other_cell.seen_time_keys:
            for pair in other_cell.seen_time_keys[group]:
                if pair not in self.seen_time_keys[group]:
                    self.seen_time_keys[group].add(pair)
                    self.timegroup_key_counts[group][pair[0]] += 1

    @staticmethod
    def get_time_block(timestamp: datetime) -> str:
        """Map time to a broad block of the day."""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return '00-08'
        elif 8 <= hour < 12:
            return '08-12'
        elif 12 <= hour < 16:
            return '12-16'
        elif 16 <= hour < 20:
            return '16-20'
        else:
            return '20-24'

    def to_dict(self) -> dict:
        """Optional: convert to serializable format if needed."""
        return {
            'i': self.i,
            'j': self.j,
            'directions_by_time': {
                group: {k: dict(c) for k, c in group_data.items()}
                for group, group_data in self.directions_by_time.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Cell':
        """Reconstruct Cell from serialized dict."""
        cell = cls(data['i'], data['j'])
        for group, group_data in data['directions_by_time'].items():
            for k, c_dict in group_data.items():
                cell.directions_by_time[group][k] = Counter(c_dict)
        return cell