import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict, Counter
from scipy.special import rel_entr
from path_finder import get_flowmap_pkl_paths
from tqdm import tqdm
from flowmap import FlowMap
from cell import Cell
from plotter import plot_kl_distributions_by_timegroup, plot_cell_distribution_comparison

def load_flowmap(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def kl_divergence(p, q, epsilon=1e-4):
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(rel_entr(p, q))

def volume_rd(vol_tr_total, vol_obs_total, n_tr, n_obs):
    vol_tr_avg = vol_tr_total / n_tr if n_tr > 0 else 0.0
    vol_obs_avg = vol_obs_total / n_obs if n_obs > 0 else 0.0

    if max(vol_tr_avg, vol_obs_avg) > 0:
        epsilon = 1e-6
        vol_rd = math.log((vol_obs_avg + epsilon) / (vol_tr_avg + epsilon))
    else:
        vol_rd = 0.0
    
    return vol_tr_avg, vol_obs_avg, vol_rd

def confidence_weighted_metrics(group, dir_kl, vel_kl, vol_rd, vol_obs_total, vol_tr_total):
    volume_th = 1e-5
    if group == 'hourly':
        volume_th = 101
    if group == 'time_block':
        volume_th = 201
    if group == 'weekday':
        volume_th = 1001
    
    confidence = min(vol_obs_total / volume_th,
                     vol_tr_total / volume_th,
                     1.0)
    
    if vol_obs_total < 10:
        confidence = 0
    
    dir_kl_w = dir_kl * confidence ** 2
    vel_kl_w = vel_kl * confidence ** 2
    vol_rd_w = vol_rd * confidence ** 2

    return dir_kl_w, vel_kl_w, vol_rd_w

def kl_divergence_between_flowmaps(trained, observed, time_groups=('hourly', 'time_block', 'weekday', 'daily'), velocity_bins=24, velocity_range=(0, 3.0)):
    results = []

    for (i, j), cell_tr in tqdm(trained.grid.items(), desc='Calculating KL-Divergence between base model and new observations'):
        cell_obs = observed.grid.get((i, j))
        if cell_obs is None:
            continue

        for group in time_groups:
            keys_tr = set(cell_tr.directions_by_time.get(group, {}).keys())
            keys_obs = set(cell_obs.directions_by_time.get(group, {}).keys())
            shared_keys = keys_tr & keys_obs

            for key in shared_keys:
                # --- Direction KL-Divergence ---
                dist_tr = cell_tr.get_direction_distribution(group, key)
                dist_obs = cell_obs.get_direction_distribution(group, key)
                dir_p = [dist_obs[d] for d in cell_tr.ALL_DIRECTIONS]
                dir_q = [dist_tr[d] for d in cell_tr.ALL_DIRECTIONS]
                dir_kl = kl_divergence(dir_p, dir_q)

                # --- Velocity KL-Divergence ---
                vel_p = cell_obs.get_velocity_histogram(group, key, bins=velocity_bins, range=velocity_range)
                vel_q = cell_tr.get_velocity_histogram(group, key, bins=velocity_bins, range=velocity_range)
                vel_kl = kl_divergence(vel_p, vel_q)

                # --- Volume Relative difference ---
                vol_tr_total = cell_tr.get_volume(group, key)
                vol_obs_total = cell_obs.get_volume(group, key)
                n_tr = cell_tr.get_timegroup_key_count(group, key)
                n_obs = cell_obs.get_timegroup_key_count(group, key)
                vol_tr_avg, vol_obs_avg, vol_rd = volume_rd(vol_tr_total, vol_obs_total, n_tr, n_obs)

                dir_kl_w, vel_kl_w, vol_rd_w = confidence_weighted_metrics(group, dir_kl, vel_kl, vol_rd, vol_obs_total, vol_tr_total)

                #if (i == 152 and j == -159):
                #    print("-----------")
                #    print(f"raw trained: {vol_tr_total}, cell: {i,j}, group: {group}, key: {key}")
                #    print(f"raw observed: ", vol_obs_total, group, key)
                #    print(f"group counts trained: ", n_tr)
                #    print(f"group counts observed: ", n_obs)
                #    print(f"avg trained: ", vol_tr_avg)
                #    print(f"avg observed: ", vol_obs_avg)
                #    print("-----------")

                results.append({
                    'cell': (i, j),
                    'time_group': group,
                    'key': key,
                    'direction_kl': dir_kl,
                    'direction_kl_w': dir_kl_w,
                    'velocity_kl': vel_kl,
                    'velocity_kl_w': vel_kl_w,
                    'volume_rd': vol_rd,
                    'volume_rd_w': vol_rd_w,
                    'volume_expected_tr': vol_tr_avg,
                    'volume_expected_obs': vol_obs_avg,
                })

    return results

def main():
    # ------------------- CONFIG ---------------------
    t = 7   # trained flowmap file index in directory
    o = 4   # observed flowmap file index in directory
    direction_threshold = 0.3
    velocity_threshold = 0.35
    volume_threshold = 0.8    # Log-scale ratio. +0.7 = Observed volume 2x compared to expected volume
                                #                  -0.7 = Observed volume 0.5x compared to expected volume
    # ------------------------------------------------

    flowmap_path_list = get_flowmap_pkl_paths()
    flowmap_path_trained = flowmap_path_list[t]
    flowmap_path_observed = flowmap_path_list[o]

    print("Loading flowmap files...")
    print(f"Trained map path: {flowmap_path_trained}")
    print(f"Observed map path: {flowmap_path_observed}")
    trained_map = load_flowmap(flowmap_path_trained)
    observed_map = load_flowmap(flowmap_path_observed)

    print("Computing KL divergences...")
    results = kl_divergence_between_flowmaps(trained_map, observed_map)
    abnormal_results = [r for r in results if (
        r['direction_kl_w'] > direction_threshold or
        r['velocity_kl_w'] > velocity_threshold or
        r['volume_rd_w'] > volume_threshold
    )]
    #plot_kl_distributions_by_timegroup(results)

    print(f"Total of {len(results)} cell instances between trained and observed flowmaps were compared across different time slots.")
    print(f"{len(abnormal_results)} of those were abnormal.")
    print("\nVisualizing abnormal cells...")
    

    for r in abnormal_results[:100]:
        i, j = r['cell']
        group = r['time_group']
        key = r['key']
        trained_cell = trained_map.grid[(i, j)]
        observed_cell = observed_map.grid[(i, j)]

        print(f"Visualizing cell {i},{j} — [{group} = {key}] "
              f"Dir KL={r['direction_kl_w']:.3f}, Vel KL={r['velocity_kl_w']:.3f}, Vol RD={r['volume_rd_w']:.3f}")
        plot_cell_distribution_comparison(trained_cell, observed_cell, time_group=group, key=key, results=abnormal_results)

if __name__ == "__main__":
    main()