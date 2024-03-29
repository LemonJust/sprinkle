"""
Second script in the analysis pipeline:
1. save_transform_class_centroids.py
2. [this one] assign_synapses_to_neurons.py
3. analyze_assignments.py
Given three csv files, two with the synapses (tp1 and tp2), one with the neurons, assign each synapse to a neuron based on the distance between them using kd-trees.

Input csv have the following columns:
- synapses: centroid_id, x, y, z
- neurons: centroid_id, x, y, z

The output csv should have the following columns:
- synapse_id, neuron_id

PROBLEM:
    There is a slight problem with this approach : the synapses are assigned to the closest nuclei, which could turn out to be a glial cell. In the future - we should remove the glial cells from the nuclei csv file before running the assignment step.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from scipy.spatial import cKDTree

def assign_synapses_to_neurons(synapses_file: str | Path, 
                               neurons_file: str | Path, 
                               out_file: str | Path):
    """
    Assign each synapse to the closest neuron and calculate the distance between them.

    Args:
        synapses_file: path to the synapses csv file
        neurons_file: path to the neurons csv file
        distance_threshold: the distance threshold to use for assignment
        out_file: path to save the output csv file
    """
    synapses = pd.read_csv(synapses_file)
    neurons = pd.read_csv(neurons_file)

    # create the kd-tree for the neurons
    tree = cKDTree(neurons[['x', 'y', 'z']])
    # query the tree for each synapse
    distances, indices = tree.query(synapses[['x', 'y', 'z']], k=1)

    # grab the neuron ids
    neuron_id = [neurons.iloc[index]['centroid_id'] for index in indices]
    
    # assign the synapses to the neurons
    synapses['neuron_id'] = neuron_id
    synapses['neuron_x'] = [neurons.iloc[index]['x'] for index in indices]
    synapses['neuron_y'] = [neurons.iloc[index]['y'] for index in indices]
    synapses['neuron_z'] = [neurons.iloc[index]['z'] for index in indices]
    synapses['distance'] = distances

    # save the output
    synapses.to_csv(out_file, index=False)
    return synapses

def threshold_synapses(synapses: pd.DataFrame, 
                       distance_threshold: float, 
                       out_file: str | Path):
    """
    Threshold the synapses based on the distance to the neurons.

    Args:
        synapses: the synapses dataframe with the distance column
        distance_threshold: the distance threshold to use for assignment
        out_file: path to save the output csv file
    """
    # threshold the synapses
    synapses = synapses[synapses['distance'] < distance_threshold]

    # save the output
    synapses.to_csv(out_file, index=False)

def process_tp(neurons_file: Path,
                synapses_file: Path,
                distance_file: Path):

    synapses = assign_synapses_to_neurons(synapses_file, neurons_file, distance_file)

    thresholds = [10,5,3,2.5,2]

    for distance_threshold in thresholds:
        out_file = distance_file.with_name(distance_file.stem + 
                                f'_thresholded_{distance_threshold}um.csv')
        threshold_synapses(synapses, distance_threshold, out_file)

    # process the pixel file:
    # grab original file with pixel coordinates - replace_um with _pix
    pixel_file = synapses_file.with_name(synapses_file.stem.replace('um', 'pix') + '.csv')
    synapses_pix = pd.read_csv(pixel_file)
    # add distance to pixel file
    synapses_pix['neuron_id'] = synapses['neuron_id']
    synapses_pix['distance'] = synapses['distance']
    # swap um to pix in the distance file
    assert "_um" in distance_file.stem, "distance file should have um in the name to replace with pix"
    distance_file = distance_file.with_name(distance_file.stem.replace('um', 'pix') + '.csv')
    synapses_pix.to_csv(distance_file, index=False)

    # threshold the synapses in the pixel file
    for distance_threshold in thresholds:
        out_file = distance_file.with_name(distance_file.stem + 
                                f'_thresholded_{distance_threshold}um.csv')
        threshold_synapses(synapses_pix, distance_threshold, out_file)

def process_test():
    sample_path = Path('/mnt/sprinkle/datasets/2024/Subject_1-26MC/')
    neurons_file = sample_path /'Centroids_1-26NA_1-26NA_fixed_channel3_hoechst_labels_st1_inhibitory_neuron_ct1_SyN_multiLabel_tt5_um.csv'

    assignment_path = sample_path / 'assignments'
    assignment_path.mkdir(exist_ok=True)

    # process tp1
    synapses_file = sample_path /'Centroids_1-26N4_1-26N4_live_tp1_labels_st2_SyN_multiLabel_tt6_um.csv'
    distance_file = assignment_path /'Distance_tp1_synapses_to_neurons_um.csv'
    process_tp(neurons_file, synapses_file, distance_file)

    # process tp2
    synapses_file = sample_path /'Centroids_1-26N6_1-26N6_live_tp2_um.csv'
    distance_file = assignment_path /'Distance_tp2_synapses_to_neurons_um.csv'
    process_tp(neurons_file, synapses_file, distance_file)

def proces_sample(sample: str):
    # sample = "1-26MC"
    centroid_path = Path(f'/mnt/sprinkle/datasets/2024/Subject_{sample}/centroids')
    neurons_file = centroid_path / 'nuclei_in_tp2_um.csv'

    assignment_path = centroid_path / 'assignments'
    assignment_path.mkdir(exist_ok=True)

    # process tp1
    synapses_file = centroid_path / 'tp1_synapses_in_tp2_um.csv'
    distance_file = assignment_path /'Distance_tp1_synapses_to_neurons_um.csv'
    process_tp(neurons_file, synapses_file, distance_file)

    # process tp2
    synapses_file = centroid_path /'tp2_synapses_um.csv'
    distance_file = assignment_path /'Distance_tp2_synapses_to_neurons_um.csv'
    process_tp(neurons_file, synapses_file, distance_file)

if __name__ == "__main__":
    proces_sample("1-26WM")
    



