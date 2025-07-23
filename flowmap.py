import pickle
from cell import Cell
from datetime import datetime
from path_finder import get_trajectory_smoothed_pkl_paths, save_flowmap_pkl_path, get_flowmap_pkl_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

    
def load_trajectory_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_trajectory_dict(trajectory_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory_dict, f)

def debug_plot_fitted_direction(traj, entry_index, exit_index, fit_range=3):

    # Get fitting window
    start = max(0, entry_index - fit_range)
    end = min(len(traj), exit_index + fit_range + 1)
    context = traj[start:end]

    # Extract x, y
    x = [float(p[1]) for p in context]
    y = [float(p[2]) for p in context]

    # Fit line
    if len(set(x)) == 1:
        slope = None
        angle_deg = 90.0 if y[-1] > y[0] else 270.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        angle_rad = math.atan(slope)
        angle_deg = math.degrees(angle_rad)
        if x[-1] < x[0]:
            angle_deg += 180

    direction = angle_to_direction(angle_deg)
    dx, dy = direction

    # Plot raw segment
    plt.figure(figsize=(7, 6))
    plt.plot(x, y, 'o-', label='Trajectory segment')

    # Plot fitted line
    if slope is not None:
        x_fit = np.linspace(min(x), max(x), 2)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'r--', label=f'Fitted line ({angle_deg:.1f}°)')
    else:
        plt.axvline(x[0], color='r', linestyle='--', label=f'Vertical line ({angle_deg:.1f}°)')

    # Plot direction arrow at center
    x_mid = sum(x) / len(x)
    y_mid = sum(y) / len(y)
    plt.arrow(x_mid, y_mid, dx * 0.5, dy * 0.5, color='green', head_width=0.3, label=f'Direction {dx, dy}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitted Direction Debug View')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def fit_direction(points):
    if len(points) < 2:
        return None
    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    if len(set(x)) == 1:
        # Perfect vertical line
        angle_deg = 90.0 if y[-1] > y[0] else 270.0
    else:
        slope, _ = np.polyfit(x, y, 1)
        angle_rad = math.atan(slope)
        angle_deg = math.degrees(angle_rad)
        if x[-1] < x[0]:
            angle_deg += 180  # flip for direction
    return angle_to_direction(angle_deg)

def angle_to_direction(theta_degrees):
    # Normalize to [0, 360)
    angle = theta_degrees % 360
    direction_vectors = {
        'E':  (1, 0),
        'NE': (1, 1),
        'N':  (0, 1),
        'NW': (-1, 1),
        'W':  (-1, 0),
        'SW': (-1, -1),
        'S':  (0, -1),
        'SE': (1, -1),
    }
    direction_bins = {
        'E':   (337.5, 22.5),
        'NE':  (22.5, 67.5),
        'N':   (67.5, 112.5),
        'NW':  (112.5, 157.5),
        'W':   (157.5, 202.5),
        'SW':  (202.5, 247.5),
        'S':   (247.5, 292.5),
        'SE':  (292.5, 337.5),
    }
    for dir_label, (low, high) in direction_bins.items():
        if low < high:
            if low <= angle < high:
                return direction_vectors[dir_label]
        else:  # Wraparound case for 'E'
            if angle >= low or angle < high:
                return direction_vectors[dir_label]
    return None


class FlowMap:
    def __init__(self, cell_size=0.5):
        self.cell_size = cell_size
        self.grid = {}  # (i, j) → Cell

    def get_cell_coords(self, x, y):
        """Convert real-world (x, y) to discrete grid cell indices (i, j)."""
        return int(x // self.cell_size), int(y // self.cell_size)

    def get_or_create_cell(self, i, j):
        """Return existing Cell or create a new one."""
        if (i, j) not in self.grid:
            self.grid[(i, j)] = Cell(i, j)
        return self.grid[(i, j)]
    
    def update_cell_direction(self, cell, traj, entry_index, exit_index):
        """
        Use linear fit to determine direction through a cell based on
        points before, inside, and after the cell.
        """
        start = max(0, entry_index - 3)
        end = min(len(traj), exit_index + 3)
        context_points = [
            (float(traj[j][1]), float(traj[j][2]))
            for j in range(start, end)
        ]
        direction = fit_direction(context_points)
        if direction is not None:
            cell.update_direction(traj[entry_index][0], direction[0], direction[1])

    def update_cell_velocity(self, cell, traj, entry_index, exit_index):
        start_idx = max(0, entry_index - 3)
        end_idx = min(len(traj), exit_index + 3)

        path = traj[start_idx:end_idx + 1]
        if len(path) < 2:
            return

        # Total distance traveled through this segment
        total_distance = 0.0
        for i in range(1, len(path)):
            _, x1, y1 = path[i - 1]
            _, x2, y2 = path[i]
            total_distance += math.hypot(x2 - x1, y2 - y1)

        t_start = path[0][0]
        t_end = path[-1][0]
        delta_t = (t_end - t_start).total_seconds()
        if delta_t > 0:
            avg_speed = total_distance / delta_t
            cell.update_velocity(t_start, avg_speed)

    def update_cell_volume(self, cell, timestamp):
        cell.update_volume(timestamp)

    def update_with(self, other_flowmap):
        for (i, j), other_cell in other_flowmap.grid.items():
            if (i, j) in self.grid:
                self.grid[(i, j)].update_with(other_cell)
            else:
                # Deep copy the other_cell
                new_cell = Cell(i, j)
                new_cell.update_with(other_cell)
                self.grid[(i, j)] = new_cell

    def process_trajectory(self, trajectory):

        if len(trajectory) < 2:
            return

        # Convert to float and datetime
        traj = []
        for t, x, y in trajectory:
            if isinstance(t, str):
                t = datetime.fromisoformat(t)
            traj.append((t, float(x), float(y)))

        prev_cell = None
        cell_points = []
        entry_index = None

        for i in range(len(traj)):
            t, x, y = traj[i]
            i_cell, j_cell = self.get_cell_coords(x, y)

            if prev_cell is None:
                prev_cell = (i_cell, j_cell)
                entry_index = i
                cell_points = [(x, y)]
                continue

            if (i_cell, j_cell) == prev_cell:
                cell_points.append((x, y))
            else:
                cell = self.get_or_create_cell(*prev_cell)
                self.update_cell_direction(cell, traj, entry_index, i)
                self.update_cell_velocity(cell, traj, entry_index, i)
                self.update_cell_volume(cell, traj[entry_index][0])

                # Reset tracking for new cell
                prev_cell = (i_cell, j_cell)
                entry_index = i
                cell_points = [(x, y)]

        # Handle the final segment (in case the trajectory ends inside a cell)
        if cell_points:
            cell = self.get_or_create_cell(*prev_cell)
            self.update_cell_direction(cell, traj, entry_index, len(traj) - 1)
            self.update_cell_velocity(cell, traj, entry_index, len(traj) - 1)
            self.update_cell_volume(cell, traj[entry_index][0])

    def train_from_trajectories(self, trajectory_dict):
        """
        Train the flow map from a dictionary of trajectories.
        Format: {agent_id: [(t, x, y), ...]}
        """
        for agent_id, traj in tqdm(trajectory_dict.items(), desc="Training cells"):
            self.process_trajectory(traj)

    def get_cell(self, x, y):
        """Access a cell by real-world coordinates."""
        i, j = self.get_cell_coords(x, y)
        return self.grid.get((i, j), None)

    def get_all_cells(self):
        """Returns all Cell objects."""
        return self.grid.values()
    
    
def main():
    # ------------ CONFIG ------------
    cell_size = 1.00                        # Cell size for the created flow map
    i = 0                                   # List element for file selection (0-5) (MODE 1)
    week_no = i                             # Week number for file saving/naming (MODE 1)
    mode = 2                                # Mode 1 = Train individual blocks of data (weeks)
                                            # Mode 2 = Combine multiple flowmaps into one
    # --------------------------------
    for i in range(6):
        if mode == 1:
            input_path_list = get_trajectory_smoothed_pkl_paths()
            input_path = input_path_list[i]
            output_path = save_flowmap_pkl_path(i)

            # LOAD
            print(f"Loading trajectory data from {input_path}...")
            traj_data = load_trajectory_dict(input_path)

            # Train the flow map
            print("Starting cell training...")
            flowmap = FlowMap(cell_size=cell_size)
            flowmap.train_from_trajectories(traj_data)

            # Save
            print("Saving flowmap...")
            save_trajectory_dict(flowmap, output_path)

    if mode == 2:
        flowmap_path_list = get_flowmap_pkl_paths()
        output_path = save_flowmap_pkl_path('012345')

        flowmap_combined = load_trajectory_dict(flowmap_path_list[0])
        for path in tqdm(flowmap_path_list[1:6], desc='Combining maps...'):
            print(f"Adding: {path} to the flowmap.")
            next_week = load_trajectory_dict(path)
            flowmap_combined.update_with(next_week)

        # Save
        print("Saving flowmap...")
        save_trajectory_dict(flowmap_combined, output_path)


if __name__ == "__main__":
    main()