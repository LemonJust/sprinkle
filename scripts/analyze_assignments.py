"""
Last script in the analysis pipeline:
1. save_transform_class_centroids.py
2. assign_synapses_to_neurons.py
3. [this one] analyze_assignments.py

IMPORTANT:
    1. before running this, make sure that you put your classification results to the correct folder:
        - for neuron-based classiffication: sample_folder/classification/inhibitory_and_neuron.csv
        - for glia-based classification: sample_folder/classification/inhibitory_and_glia.csv
    2. you might also have to change the column titles in the csv file:
        - ID to neuron_id (or make sure there is a column neuron_id)
        - Class to class (or make sure there is a column class)
        - Subclass to subclass (or make sure there is a column subclass)
        Sorry! :(

Exploratory data analysis of the synapse to neuron assignments.
"""
from __future__ import annotations
from typing import Literal
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

def calculate_number_of_synapses_per_neuron(synapses_file: str | Path, 
                                            distance_threshold: float):
    """
    Calculate the number of synapses per neuron for a given thresholded distance.

    Args:
        synapses_file: path to the synapses csv file with all the distances
            (~ Distance_tp1_synapses_to_neurons_um.csv)
    """
    synapses = pd.read_csv(synapses_file)
    # threshold the synapses
    synapses = synapses[synapses['distance'] < distance_threshold]

    # save thresholded synapses
    synapses.to_csv(synapses_file.with_name(synapses_file.stem + f'_threshold_distance_{distance_threshold}um.csv'), index=False)

    synapses_per_neuron = synapses['neuron_id'].value_counts()
    return synapses_per_neuron

def calculate_synapse_change_per_neuron(synapses_before_file: str | Path,
                                        synapses_after_file: str | Path,
                                        distance_threshold: float):
    """
    Calculate the number of synapses per neuron for a given thresholded distance.

    Args:
        synapses_before_file: path to the synapses csv file with all the distances
            (~ Distance_tp1_synapses_to_neurons_um.csv)
        synapses_after_file: path to the synapses csv file with all the distances
            (~ Distance_tp2_synapses_to_neurons_um.csv)
    """
    synapses_per_neuron_before = calculate_number_of_synapses_per_neuron(
                                            synapses_before_file,
                                            distance_threshold)
    synapses_per_neuron_after = calculate_number_of_synapses_per_neuron(
                                            synapses_after_file,
                                            distance_threshold)
    # combine the two series
    synapses_per_neuron = pd.concat([synapses_per_neuron_before,
                                    synapses_per_neuron_after],
                                    axis=1,
                                    keys=['before', 'after'])
    synapses_per_neuron.fillna(0, inplace=True)

    # calculate the change
    synapses_per_neuron['change'] = synapses_per_neuron['after'] - synapses_per_neuron['before']

    # Reset the index of the DataFrame
    synapses_per_neuron = synapses_per_neuron.reset_index()

    return synapses_per_neuron

def filter_by_classes(synapses_per_neuron: pd.DataFrame, 
                        classes_file: str | Path):
        """
        Filter the synapses per neuron dataframe by the classes.
    
        Args:
            synapses_per_neuron: the dataframe with the number of synapses per neuron
                and the change
            classes_file: the file with the classes for the neurons 
                (the pixel values of the mask of each neuron in the image corresponds to the neuron id in the dataframe)
        """
        classes = pd.read_csv(classes_file)
        synapses_per_neuron = synapses_per_neuron.join(classes.set_index('neuron_id'), on='neuron_id')
        return synapses_per_neuron

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
    plt.title('Change in synapses per neuron')
    plt.show()

# def make_class_image(synapses_per_neuron: pd.DataFrame,
#                         labels_image_file: str | Path,
#                         image_file: str | Path, 
#                         neuron_class: str = None,
#                         neuron_subclass: str = None):
#         """
#         Create an image of the number of synapses per neuron.
    
#         Args:
#             synapses_per_neuron: the dataframe with the number of synapses per neuron
#                 and the change
#             labels_image_file: the file with the labels for the neurons 
#                 (the pixel values of the mask of each neuron in the image corresponds to the neuron id in the dataframe)
#             image_file: the file to save the image to
#             neuron_class: the class to filter the neurons
#             neuron_subclass: the subclass to filter the neurons
#         """
#         if neuron_class:
#             synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['class'] == neuron_class]
#         if neuron_subclass:
#             synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['subclass'] == neuron_subclass]
    
#         # read the labels image
#         labels_image = tif.imread(labels_image_file)
#         # create an empty image
#         class_image = np.zeros_like(labels_image, dtype=np.uint8)
#         # keep neurons that had any synapses assigned either before or after
#         synapses_per_neuron = synapses_per_neuron[(synapses_per_neuron['before'] > 0) | (synapses_per_neuron['after'] > 0)]
#         # fill the image with the neuron ids
#         for neuron_id in synapses_per_neuron['neuron_id']:
#             class_image[labels_image == neuron_id] = 1

#         # save the image
#         tif.imsave(image_file, class_image)

def create_change_image(synapses_per_neuron: pd.DataFrame,
                        labels_image_file: str | Path,
                        save_image_file: str | Path, 
                        neuron_class: str = None,
                        neuron_subclass: str = None):
    """
    Create an image of the change in synapses per neuron.

    Args:
        synapses_per_neuron: the dataframe with the number of synapses per neuron
            and the change
        labels_image_file: the file with the labels for the neurons 
            (the pixel values of the mask of each neuron in the image corresponds to the neuron id in the dataframe)
        save_image_file: the file to save the image to
        neuron_class: the class to filter the neurons
        neuron_subclass: the subclass to filter the neurons
    """
    if neuron_class:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['class'] == neuron_class]
    if neuron_subclass:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['subclass'] == neuron_subclass]
    print("synapses_per_neuron after filtering:")
    print(synapses_per_neuron.head())

    # read the labels image
    labels_image = tif.imread(labels_image_file)
    # create an empty image
    posintive_change_image = np.zeros_like(labels_image, dtype=np.uint8)
    negative_change_image = np.zeros_like(labels_image, dtype=np.uint8)
    unchanged_image = np.zeros_like(labels_image, dtype=np.uint8)

    # Fill the image with the change in synapses
    for neuron_id, change in zip(synapses_per_neuron['neuron_id'], synapses_per_neuron['change']):
        if change > 0:
            posintive_change_image[labels_image == neuron_id] = change
        elif change < 0:
            negative_change_image[labels_image == neuron_id] = -change
        else:
            unchanged_image[labels_image == neuron_id] = 1
    
    # save the images
    tif.imwrite(save_image_file.with_name(save_image_file.stem + '_positive_change.tif'), posintive_change_image)
    tif.imwrite(save_image_file.with_name(save_image_file.stem + '_negative_change.tif'), negative_change_image)
    tif.imwrite(save_image_file.with_name(save_image_file.stem + '_unchanged.tif'), unchanged_image)

def create_class_csv(synapses_per_neuron: pd.DataFrame,
                      nuclei_points_file: str | Path,
                      neuron_class: str = None,
                      neuron_subclass: str = None):
    """
    Creates the scv files for the synapses_per_neuron and the nuclei centroids filtered by the classes.
    Filters the synapses_per_neuron dataframe by the classes and then filters the nuclei points by the neurons that had synapses assigned to them.
    """
    if neuron_class:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['class'] == neuron_class]
    if neuron_subclass:
        synapses_per_neuron = synapses_per_neuron[synapses_per_neuron['subclass'] == neuron_subclass]

    print(f"synapses_per_neuron after filtering by class {neuron_class} and subclass {neuron_subclass}:")
    print(synapses_per_neuron['change'].describe())

    # save filtered synapses_per_neuron
    synapses_per_neuron.to_csv(nuclei_points_file.parent / f'synapses_per_neuron_{neuron_class}_{neuron_subclass}.csv',       index=False)
    
    # filter the nuclei points by the neurons that had synapses assigned to them
    synapses_per_neuron = synapses_per_neuron[(synapses_per_neuron['before'] > 0) | (synapses_per_neuron['after'] > 0)]
    # save the filtered nuclei points
    nuclei_points = pd.read_csv(nuclei_points_file)
    nuclei_points = nuclei_points[nuclei_points['centroid_id'].isin(synapses_per_neuron['neuron_id'])]

    # save the scv files
    nuclei_points.to_csv(nuclei_points_file.with_name(nuclei_points_file.stem + f'_assigned_{neuron_class}_{neuron_subclass}.csv'), index=False)

def modify_class_df_after_glia_classification(class_file: str | Path) -> pd.DataFrame:
    """
    class_file includes the following columns:
    - neuron_id, class, subclass
    
    Currently there are three classes:
    - neuron
    - glia 
    - nuclei

    This function replaces all entries that have class 'nuclei' with the class 'neuron' and subclass 'excitatory'. Assuming that the classifier only explicitly labeled the inhibitory neurons and glia.

    Args:
        class_file: the file with the classes for the neurons.
    
    Returns:
        the modified dataframe
    """
    classes = pd.read_csv(class_file)
    classes.loc[classes['class'] == 'nuclei', ['class', 'subclass']] = ['neuron', 'excitatory']

    # save to the same folder as the original file as "inhibitory_excitatory_glia.csv"
    class_file = Path(class_file)
    classes.to_csv(class_file.parent/ "inhibitory_excitatory_glia.csv", index=False)

    return classes

def modify_class_df_after_neuron_classification(class_file: str | Path) -> pd.DataFrame:
    """
    class_file includes the following columns:
    - neuron_id, class, subclass
    
    Currently there are three classes:
    - neuron
    - glia 
    - nuclei

    This function replaces all entries that have class 'neuron' and no subclass with the class 'neuron' and subclass 'excitatory', and all entries that have class 'nuclei' with the class 'glia'. Assuming that the classifier only explicitly labeled the inhibitory neurons and neurons.

    Args:
        class_file: the file with the classes for the neurons.
    
    Returns:
        the modified dataframe
    """
    classes = pd.read_csv(class_file)
    classes.loc[classes['class'] == 'nuclei', ['class', 'subclass']] = ['glia', None]
    classes.loc[(classes['class'] == 'neuron') & (classes['subclass'].isnull()), 'subclass'] = 'excitatory'

    # save to the same folder as the original file as "inhibitory_excitatory_glia.csv"
    class_file = Path(class_file)
    classes.to_csv(class_file.parent/ "inhibitory_excitatory_glia.csv", index=False)

    return classes

def add_classes_to_distance_df(synapses_df: pd.DataFrame, 
                               classes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the classes to the synapses dataframe.

    Args:
        synapses_df: the dataframe with the synapses
        classes_df: the dataframe with the classes for the neurons

    Returns:
        the modified dataframe
    """
    synapses_df = synapses_df.join(classes_df.set_index('neuron_id'), on='neuron_id')
    return synapses_df

def process_test():
        # synapses_file_before = Path("D:/Code/repos/sprinkle/datasets/2024/Subject_1-26MC/assignments/Distance_tp1_synapses_to_neurons_um.csv")
    # synapses_file_after = Path("D:/Code/repos/sprinkle/datasets/2024/Subject_1-26MC/assignments/Distance_tp2_synapses_to_neurons_um.csv")

    # # plot_synapse_to_neuron_distance_histogram(synapses_file)

    # distance_threshold = 5
    # print(
    #     f"Number of synapses per neuron for distance threshold of {distance_threshold}um:")
    # synapses_per_neuron = calculate_synapse_change_per_neuron(
    #     synapses_file_before, synapses_file_after, distance_threshold)
    # # print(synapses_per_neuron)

    # # save_synapse_change_per_neuron(synapses_file_before, synapses_file_after, distance_threshold)

    # # plot_synapse_change_per_neuron(synapses_per_neuron)
    # print(synapses_per_neuron['change'].describe())

    # plot_synapse_change_per_neuron(synapses_per_neuron)

    synapses_file_before = Path(
        "/mnt/sprinkle/datasets/2024/Subject_1-26XY/centroids/assignments/Distance_tp1_synapses_to_neurons_pix.csv")
    synapses_file_after = Path(
        "/mnt/sprinkle/datasets/2024/Subject_1-26XY/centroids/assignments/Distance_tp2_synapses_to_neurons_pix.csv")
    neuron_classes_file = Path(
        "/mnt/sprinkle/datasets/2024/Subject_1-26XY/classification/inhibitory_and_glia.csv")
    class_df = modify_class_df_after_neuron_classification(neuron_classes_file)

    label_image_file = Path(
        "/mnt/sprinkle/datasets/2024/Subject_1-26XY/fixed_hoechst_copy_labels_st2_SyNRA_multiLabel_tt5.tif")

    # plot_synapse_to_neuron_distance_histogram(synapses_file)

    distance_threshold = 5
    print(
        f"Number of synapses per neuron for distance threshold of {distance_threshold}um:")
    synapses_per_neuron = calculate_synapse_change_per_neuron(
        synapses_file_before, synapses_file_after, distance_threshold)
    print("Before adding classes:")
    print(synapses_per_neuron.head())

    synapses_per_neuron = add_classes_to_distance_df(synapses_per_neuron, class_df)
    
    # print(synapses_per_neuron)

    # save_synapse_change_per_neuron(synapses_file_before, synapses_file_after, distance_threshold)

    print(synapses_per_neuron['change'].describe())

    # plot_synapse_change_per_neuron(synapses_per_neuron)

    neuron_classes_file = Path('/mnt/sprinkle/datasets/2024/Subject_1-26XY/centroids/nuclei_in_tp2_pix.csv')
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='neuron', neuron_subclass='inhibitory')
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='neuron', neuron_subclass='excitatory')
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='glia')

    # create change_images ______________________

    # inhibitory neurons
    save_image_file = label_image_file.parent / 'fixed_hoechst_labels_SyNRA_multiLabel_inhibitory.tif'
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='neuron',
                        neuron_subclass='inhibitory')
    # excitatory neurons
    save_image_file = label_image_file.parent / 'fixed_hoechst_labels_SyNRA_multiLabel_excitatory.tif'
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='neuron',
                        neuron_subclass='excitatory')
    # glia
    save_image_file = label_image_file.parent / 'fixed_hoechst_labels_SyNRA_multiLabel_glia.tif'
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='glia')

def process_sample(sample: str, 
                   hoechst_in_tp2_label_image_filename: str, 
                   classification_type = Literal['neuron', 'glia']) -> None:
    # sample = "1-26MC"
    # hoechst_in_tp2_label_image_filename = "fixed_hoechst_copy_labels_st2_SyNRA_multiLabel_tt5.tif"
    sample_path = Path(f'/mnt/sprinkle/datasets/2024/Subject_{sample}/')
    synapses_file_before = sample_path / "centroids/assignments/Distance_tp1_synapses_to_neurons_pix.csv"
    synapses_file_after = sample_path / "centroids/assignments/Distance_tp2_synapses_to_neurons_pix.csv"

    if classification_type == 'neuron':
        neuron_classes_file = sample_path / "classification/inhibitory_and_neuron.csv"
        class_df = modify_class_df_after_neuron_classification(neuron_classes_file)
    elif classification_type == 'glia':
        neuron_classes_file = sample_path / "classification/inhibitory_and_glia.csv"
        class_df = modify_class_df_after_glia_classification(neuron_classes_file)
    else:
        raise ValueError("classification_type must be either 'neuron' or 'glia'.")

    label_image_file = sample_path / hoechst_in_tp2_label_image_filename

    distance_threshold = 5
    print(
        f"Number of synapses per neuron for distance threshold of {distance_threshold}um:")
    synapses_per_neuron = calculate_synapse_change_per_neuron(
        synapses_file_before, synapses_file_after, distance_threshold)
    print("Before adding classes:")
    print(synapses_per_neuron.head())

    synapses_per_neuron = add_classes_to_distance_df(synapses_per_neuron, class_df)
    
    print(synapses_per_neuron['change'].describe())

    neuron_classes_file = sample_path / "centroids/nuclei_in_tp2_pix.csv"
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='neuron', neuron_subclass='inhibitory')
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='neuron', neuron_subclass='excitatory')
    create_class_csv(synapses_per_neuron, neuron_classes_file, neuron_class='glia')

    # create change_images ______________________

    # inhibitory neurons
    inhibitory_filename = Path(hoechst_in_tp2_label_image_filename).stem + '_inhibitory.tif'
    save_image_file = label_image_file.parent / inhibitory_filename
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='neuron',
                        neuron_subclass='inhibitory')
    # excitatory neurons
    excitatory_filename = Path(hoechst_in_tp2_label_image_filename).stem + '_excitatory.tif'
    save_image_file = label_image_file.parent / excitatory_filename
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='neuron',
                        neuron_subclass='excitatory')
    # glia
    glia_filename = Path(hoechst_in_tp2_label_image_filename).stem + '_glia.tif'
    save_image_file = label_image_file.parent / glia_filename
    create_change_image(synapses_per_neuron,
                        label_image_file,
                        save_image_file,
                        neuron_class='glia')

if __name__ == "__main__":
    # before running this, make sure that you put your classification results to the correct folder:
    # - for neuron-based classiffication: sample_folder/classification/inhibitory_and_neuron.csv
    # - for glia-based classification: sample_folder/classification/inhibitory_and_glia.csv
    # you might also have to change the column titles:
    # ID to neuron_id, Class to class and Subclass to subclass in the csv file :( sorry!
    process_sample("1-26WM", 
                  "fixed_HOECHST.ome_labels_st1_SyN_multiLabel_tt7.tif",
                  classification_type='neuron')


    

