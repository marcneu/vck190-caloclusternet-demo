import numpy as np
import pandas as pd
import h5py
import pyvista as pv
import os
import subprocess

def load_crystal_input_h5_file(file_path):
    """
    Load the event data from an HDF5 file.
    Here the input graph (feat_graphs) is expected to be stored.
    """
    print("Loading event data...")
    with h5py.File(file_path, 'r') as f:
        feat_graphs = f["float_feat_graphs"][:]
    energies = np.array(feat_graphs[:,:,0])
    positions = np.array([feat_graphs[:,:,4],feat_graphs[:,:,3],feat_graphs[:,:,2]]).transpose(1, 2, 0)*100
    unpadded_positions = [event[event.sum(axis=1) != 0] for event in positions]
    return energies, positions, unpadded_positions

def load_offline_input_h5_file(file_path):
    """
    Load the offline cluster data from an HDF5 file.
    Here the offline cluster information (truth_graphs) is expected to be stored.
    """
    print("Loading Offline data...")
    with h5py.File(file_path, 'r') as f:
        offline_graphs = f["truth_graphs"][:]
    offline_energies = np.array(offline_graphs[:,:,1])
    offline_positions = np.array([offline_graphs[:,:,4],offline_graphs[:,:,3],offline_graphs[:,:,2]]).transpose(1, 2, 0)*100
    # remove positions with all zeros
    offline_positions = [event[event.sum(axis=1) != 0] for event in offline_positions]
    return offline_energies, offline_positions

def load_offline_cluster_data(input_file):
    """
    Load the offline cluster data from the input file.
    This function reads the offline cluster energies and positions from the input file.
    """
    with h5py.File(input_file, 'r') as f:
        if "offline_cluster" in f:
            offline_cluster = f["offline_cluster"][:]
            offline_cluster_energies = offline_cluster[:,:,1]
            offline_cluster_positions = offline_cluster[:,:,2:]

            offline_cluster_energies = [event[~(event == 0)] for event in offline_cluster_energies]
            offline_cluster_positions = [event[~(event.all(axis=1) == 0)] for event in offline_cluster_positions]
    return offline_cluster_energies, offline_cluster_positions

def load_prediction_data(output_file):
    """
    Load the prediction data from the output file.
    This function reads the prediction energies and positions from the output file.
    """
    with h5py.File(output_file, 'r') as f:
        if "predictions" in f:
            pred_graphs = f["predictions"][:]
            predictions_energies = pred_graphs[:,:,1]
            predictions_positions = pred_graphs[:,:,2:]

            predictions_energies = [event[~(event == 0)] for event in predictions_energies]
            predictions_positions = [event[~(event.all(axis=1) == 0)]*100 for event in predictions_positions]
    return predictions_energies, predictions_positions

def load_crystal_data(input_file):
    """
    Load the crystal data from the input file.
    This function reads the crystal IDs and energies from the input file.
    """
    with h5py.File(input_file, 'r') as f:
        feat_graphs = f["crystals"][:]
    crystal_ids =  feat_graphs[:,:,1]
    crystal_energies =  feat_graphs[:,:,2] 
    crystal_ids = [event[~(event == 0)] for event in crystal_ids]
    crystal_energies = [event[~(event == 0)] for event in crystal_energies]
    return crystal_ids, crystal_energies

def load_track_data(input_file):
    """
    Load the track data from the input file.
    This function reads the track IDs and energies from the input file.
    """
    with h5py.File(input_file, 'r') as f:
        track_data = f["tracks"][:,:,:]
    return track_data

def generate_helix_track(predictions):
    """
    Generate helix tracks for multiple particles.
    predictions: np.ndarray of shape (6, N) where N is number of tracks.
    Returns: np.ndarray of shape (max_steps+1, N, 3)
    """
    max_steps = 500
    event, px, py, pz, charge, dphi = predictions
    pt = np.sqrt(px**2 + py**2)
    R = -1 * pt * 3.335 * 100 / (1.5 * charge)
    dx = px / pt
    dy = py / pt
    x0 = y0 = z0 = 0
    x_center = x0 - R * dy
    y_center = y0 + R * dx

    phi_start = np.arctan2(y0 - y_center, x0 - x_center)
    phi = np.broadcast_to(phi_start, (max_steps + 1, px.shape[0]))
    phi = phi_start[None, :]
    step_scale = 0.05
    phi = phi + np.arange(max_steps + 1).reshape(-1, 1) * (np.sign(R) * dphi * np.sign(dphi) * step_scale)

    x = x_center + np.abs(R) * np.cos(phi)
    y = y_center + np.abs(R) * np.sin(phi)
    z = z0 +  np.abs(R) * pz / pt * np.abs(phi_start - phi)

    bounds = 137
    for i in range(x.shape[1]):
        out_of_bounds = ((np.sqrt(x[:, i]**2 + y[:, i]**2) > bounds) | ((z[:, i] > 222) | (z[:, i] < -123)))
        if np.any(out_of_bounds):
            first_oob = np.argmax(out_of_bounds)
            x[first_oob:, i] = np.nan
            y[first_oob:, i] = np.nan
            z[first_oob:, i] = np.nan
    return np.stack((x, y, z), axis=-1) 
