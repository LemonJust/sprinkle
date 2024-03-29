"""
Exploratory data analysis of the synapse to neuron assignments.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tifffile as tif

def plot_synapse_to_neuron_distance_histogram(synapses_file: str | Path):
    """
    Plot the histogram of distances between synapses and neurons.

    Args:
        synapses_file: path to the synapses csv file with all the distances 
            (~ Distance_tp1_synapses_to_neurons_um.csv)
    """
    synapses = pd.read_csv(synapses_file)
    sns.histplot(synapses['distance'])
    plt.xlabel('Distance to neuron (um)')
    plt.ylabel('Number of synapses')
    plt.title('Distance to neuron for synapses')
    plt.show()

def plot_synapse_change_per_neuron(synapses_per_neuron: pd.DataFrame, 
                                   neuron_class: str = None, 
                                   neuron_subclass: str = None):
    """
    Plot the histogram of synapse change per neuron.

    Args:
        synapses_per_neuron: the dataframe with the number of synapses per neuron
            and the change
        neuron_class: the class of the neuron to filter
        neuron_subclass: the subclass of the neuron to filter
    """
    if neuron_class:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['class'] == neuron_class]
    if neuron_subclass:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['subclass'] == neuron_subclass]

    sns.histplot(synapses_per_neuron['change'])
    plt.xlabel('Change in synapses')
    plt.ylabel('Number of neurons')
    plt.title(f'Change in synapses per neuron\nClass: {neuron_class}\nSubclass: {neuron_subclass}')
    plt.show()

if __name__ == "__main__":
    # plot the histogram of distances between synapses and neurons
    # distance_file = 'D:/Code/repos/sprinkle/datasets/2024/Subject_1-26XY/centroids/assignments/Distance_tp1_synapses_to_neurons_um_thresholded_5um.csv'
    # plot_synapse_to_neuron_distance_histogram(distance_file)

    # plot the histogram of synapse change per neuron
    synapses_per_neuron = pd.read_csv('D:/Code/repos/sprinkle/datasets/2024/Subject_1-26XY/centroids/synapses_per_neuron_neuron_inhibitory.csv')
    plot_synapse_change_per_neuron(synapses_per_neuron, 'neuron', 'inhibitory')

    synapses_per_neuron = pd.read_csv('D:/Code/repos/sprinkle/datasets/2024/Subject_1-26XY/centroids/synapses_per_neuron_neuron_excitatory.csv')
    plot_synapse_change_per_neuron(synapses_per_neuron, 'neuron', 'excitatory')

    synapses_per_neuron = pd.read_csv('D:/Code/repos/sprinkle/datasets/2024/Subject_1-26XY/centroids/nuclei_in_tp2_pix_assigned_glia_None.csv')
    plot_synapse_change_per_neuron(synapses_per_neuron, 'glia', 'None')
    


    

