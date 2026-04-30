import os
import random
import math
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from custom_plots import apply_custom_plotting_flavor

facies_properties = {
    'undefined':           {'val': 0,  'color': "#ff0000", 'description': "Background/Undefined"},
    'channel_lag':         {'val': 1,  'color': "#f1970f", 'description': "Active channel fill, coarse-grained"},
    'point_bar':           {'val': 2,  'color': "#f3dd12", 'description': "Lower energy channel margins"},
    'sand_plug':           {'val': 3,  'color': "#af8f00", 'description': "Fine-grained oxbow/plug fill"},
    'crevasse_splay_core': {'val': 4,  'color': "#fffc65", 'description': "Proximal high-energy splay"},
    'crevasse_channel':    {'val': 5,  'color': "#ffd986", 'description': "Feeder channel for splays"},
    'crevasse_splay_delta':{'val': 6,  'color': "#ff9853", 'description': "Distal fan-like splay deposit"},
    'levee':               {'val': 7,  'color': "#27ae60", 'description': "Sand/silt ridges bordering channel"},
    'overbank':            {'val': 8,  'color': "#33ff00", 'description': "Stabilized/vegetated levee"},
    'mud_plug':            {'val': 9,  'color': "#fff7db", 'description': "Fine silts/clays far from channel"},
    'hemipelagic_plug':    {'val': 10, 'color': "#7a7d80", 'description': "Silts near active channel belts"},
    'wetland':             {'val': 11, 'color': "#d862f0", 'description': "Organic rich, very fine sediment"},
    'draping':             {'val': 12, 'color': "#8dd5e7", 'description': "Lateral accretion sand bodies"},
    'pelagic':             {'val': 13, 'color': "#3498db", 'description': "Lacustrine clay/silt"}
}

def create_colormap_and_legend():
    """
    Creates a ListedColormap and legend patches based on facies_properties.
    
    Returns:
        tuple: A tuple containing the custom colormap and a list of legend patches.
    """
    sorted_facies = sorted(facies_properties.items(), key=lambda item: item[1]['val'])
    color_list = [item[1]['color'] for item in sorted_facies]
    custom_cmap = ListedColormap(color_list)
    
    legend_patches = [
        mpatches.Patch(color=info['color'], label=name.replace('_', ' ').title()) 
        for name, info in sorted_facies
    ]
    return custom_cmap, legend_patches

def load_files(file):
    """
    Loads the facies array from a .npz or .npy file.
    
    Args:
        file (str): The path to the file.
    """
    if file.endswith('.npz'):
        with np.load(file) as data:
            return data['facies'] if 'facies' in data else data[data.files[0]]
    else:
        return np.load(file)


def plot_realization_slices(data_dir, output_dir, all_files, num_files, x_slice, y_slice, z_slice):
    """
    Plots orthogonal slices for a specified number of random realizations in a grid.
    
    Args:
        data_dir (str): Directory containing the .npz files.
        output_dir (str): Directory to save the resulting plots.
        all_files (list): List of all .npz file names.
        num_files (int): Number of random realizations to sample.
        x_slice (int, optional): The X-index for the vertical slice. Defaults to the middle.
        y_slice (int, optional): The Y-index for the vertical slice. Defaults to the middle.
        z_slice (int): The Z-index for the horizontal slice to plot.
    """
    print("\n--- Generating Orthogonal Slices for Multiple Realizations ---")
    num_to_sample = min(num_files, len(all_files))
    if num_to_sample == 0:
        print("No files to plot.")
        return
        
    sampled_files = random.sample(all_files, num_to_sample)
    custom_cmap, legend_patches = create_colormap_and_legend()

    # Create a grid of subplots
    fig, axes = plt.subplots(num_to_sample, 3, figsize=(18, 6 * num_to_sample), squeeze=False)

    # Get dimensions from the first file to determine slice indices if not provided
    first_file_path = os.path.join(data_dir, sampled_files[0])
    sample_data = load_files(first_file_path)
    max_z, max_y, max_x = sample_data.shape
    
    z_idx = z_slice if z_slice is not None else max_z // 2
    y_idx = y_slice if y_slice is not None else max_y // 2
    x_idx = x_slice if x_slice is not None else max_x // 2

    for i, file in enumerate(sampled_files):
        file_path = os.path.join(data_dir, file)
        data_3d = load_files(file_path)
        
        # Ensure indices are within bounds
        _max_z, _max_y, _max_x = data_3d.shape
        z_idx = min(z_idx, _max_z - 1)
        y_idx = min(y_idx, _max_y - 1)
        x_idx = min(x_idx, _max_x - 1)

        # Plot XY slice (Z-plane)
        axes[i, 0].imshow(data_3d[z_idx, :, :], cmap=custom_cmap, vmin=0, vmax=13, origin='lower', aspect='auto')
        axes[i, 0].set_ylabel(os.path.splitext(file)[0], rotation=90, size='large')
        
        # Plot XZ slice (Y-plane)
        axes[i, 1].imshow(data_3d[:, y_idx, :], cmap=custom_cmap, vmin=0, vmax=13, origin='lower', aspect='auto')

        # Plot YZ slice (X-plane)
        axes[i, 2].imshow(data_3d[:, :, x_idx], cmap=custom_cmap, vmin=0, vmax=13, origin='lower', aspect='auto')

        # Set titles only for the top row
        if i == 0:
            axes[i, 0].set_title(f"XY Plane (Z={z_idx})", fontsize=12)
            axes[i, 1].set_title(f"XZ Plane (Y={y_idx})", fontsize=12)
            axes[i, 2].set_title(f"YZ Plane (X={x_idx})", fontsize=12)

    plt.suptitle(f"Orthogonal Slices for {num_to_sample} Realizations", fontsize=16)
    fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.15, 0.5), title="Facies Types")
    plt.tight_layout(rect=[0, 0, 1.0, 0.97])

    plot_path = os.path.join(output_dir, f"orthogonal_slices_{num_to_sample}_samples.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved orthogonal slices plot to: {plot_path}")


def plot_facies_distribution(data_dir, output_dir, all_files):
    """
    Analyzes and plots the overall distribution of facies across the entire dataset.
    """
    print("\n--- Generating Overall Facies Distribution ---")
    
    # Initialize counts for 14 facies (0-13)
    total_counts = np.zeros(14, dtype=np.int64)
    num_files = len(all_files)

    for i, file in enumerate(all_files):
        filepath = os.path.join(data_dir, file)
        data = load_files(filepath)
        # Extract counts, forcing length to 14
        file_counts = np.bincount(data.ravel(), minlength=14)
        total_counts += file_counts

        if (i + 1) % 100 == 0 or (i + 1) == num_files:
            print(f"Processed {i + 1}/{num_files} files...")

    # Calculate percentages
    total_voxels = total_counts.sum()
    percentages = (total_counts / total_voxels) * 100

    # Console Output
    # print("\n--- Overall Facies Distribution Summary ---")
    # Convert facies_properties to a value-indexed map for easier lookup
    val_to_info = {info['val']: {'name': name, 'color': info['color']} 
                   for name, info in facies_properties.items()}

    # for val in range(14):
    #     if total_counts[val] > 0:
    #         name = val_to_info[val]['name'].replace('_', ' ').title()
    #         print(f"[{val:2d}] {name:<22}: {percentages[val]:6.2f}% ({total_counts[val]:,} voxels)")

    # Visualization
    plt.figure(figsize=(14, 7))
    x_values = np.arange(14)
    bar_colors = [val_to_info[val]['color'] for val in x_values]

    bars = plt.bar(x_values, percentages, color=bar_colors, edgecolor='black')

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color=val_to_info[val]['color'], 
                       label=f"{val}: {val_to_info[val]['name'].replace('_', ' ').title()}")
        for val in x_values
    ]

    plt.legend(handles=legend_patches, title="Facies Types", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(f'Combined Facies Distribution ({num_files} Samples)', fontsize=14, weight='bold')
    plt.xlabel('Facies Value', fontsize=12)
    plt.ylabel('Total Percentage (%)', fontsize=12)
    plt.xticks(x_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "overall_facies_distribution.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved distribution plot to: {plot_path}")


def plot_entropy_matrix_helper(slices_stack, slice_indices, axis_name, xlabel, ylabel, output_directory, data_dir, num_files):
    """
    Helper function to calculate and plot the entropy matrices for a given set of slices.
    
    Args:
        slices_stack (np.ndarray): 3D array of stacked slices from multiple realizations.
        slice_indices (list): List of slice indices to plot.
        axis_name (str): Name of the axis ('X', 'Y', or 'Z') being sliced.
        xlabel (str): Label for the X-axis of the plot.
        ylabel (str): Label for the Y-axis of the plot.
        output_directory (str): Directory to save the plot.
        data_dir (str): Original data directory (used for naming the output file).
        num_files (int): Number of sampled realizations.
    """
    num_realizations, _, dim_y, dim_x = slices_stack.shape
    num_facies = len(facies_properties)
    
    vmin, vmax = 0, 3.0 
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, slice_val in enumerate(slice_indices):
        probabilities = np.zeros((num_facies, dim_y, dim_x))
        for f_val in range(num_facies):
            probabilities[f_val] = np.sum(slices_stack[:, idx, :, :] == f_val, axis=0) / num_realizations

        entropy_map = entropy(probabilities, base=2, axis=0)
        im = axes[idx].imshow(entropy_map, cmap='magma', origin='lower', norm=norm)
        
        axes[idx].set_title(f"{axis_name}-Slice = {slice_val}", fontsize=12)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Entropy (bits)', rotation=270, labelpad=15)

    plane_names = {'Z': 'XY', 'Y': 'ZX', 'X': 'ZY'}
    plane = plane_names[axis_name]
    plt.suptitle(f"{plane} Plane Cell-Wise Entropy (Fixed Scale: {vmin}-{vmax} bits)\n({num_files} Realizations)", fontsize=16)

    dataset_name = os.path.basename(os.path.normpath(data_dir))
    plot_path = os.path.join(output_directory, f"entropy_matrix_{plane}_{dataset_name}_{num_files}_samples.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {plane} entropy plot to: {plot_path}")

def plot_entropy(data_dir, output_dir, all_files, num_files):
    """
    Samples a subset of files and plots the cell-wise entropy across multiple slices
    for X, Y, and Z planes to measure variability among realizations.
    
    Args:
        data_dir (str): Directory containing the .npz files.
        output_dir (str): Directory to save the resulting plots.
        all_files (list): List of all .npz file names.
        num_files (int): Number of realizations to sample.
    """
    print("\n--- Generating Entropy Matrices ---")
    num_to_sample = min(num_files, len(all_files))
    sampled_files = random.sample(all_files, num_to_sample)
    
    first_file_path = os.path.join(data_dir, sampled_files[0])
    sample_data = load_files(first_file_path)
    max_z, ny, nx = sample_data.shape
    
    random_z_slices = sorted(random.sample(range(max_z), 9))
    random_y_slices = sorted(random.sample(range(ny), 9))
    random_x_slices = sorted(random.sample(range(nx), 9))

    stack_xy = np.zeros((num_to_sample, 9, ny, nx), dtype=np.uint8) 
    stack_zx = np.zeros((num_to_sample, 9, max_z, nx), dtype=np.uint8) 
    stack_zy = np.zeros((num_to_sample, 9, max_z, ny), dtype=np.uint8) 

    valid_count = 0
    for file in sampled_files:
        try:
            data_3d = load_files(os.path.join(data_dir, file)) 
            if data_3d.size == 0 or data_3d.shape[0] == 0:
                continue
            
            stack_xy[valid_count] = data_3d[random_z_slices, :, :]
            stack_zx[valid_count] = data_3d[:, random_y_slices, :].swapaxes(0, 1)
            stack_zy[valid_count] = data_3d[:, :, random_x_slices].transpose(2, 0, 1)
            valid_count += 1
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    stack_xy = stack_xy[:valid_count]
    stack_zx = stack_zx[:valid_count]
    stack_zy = stack_zy[:valid_count]
    
    plot_entropy_matrix_helper(stack_xy, random_z_slices, 'Z', 'X', 'Y', output_dir, data_dir, valid_count)
    plot_entropy_matrix_helper(stack_zx, random_y_slices, 'Y', 'X', 'Z', output_dir, data_dir, valid_count)
    plot_entropy_matrix_helper(stack_zy, random_x_slices, 'X', 'Y', 'Z', output_dir, data_dir, valid_count)


def plot_3d_pyvista(data_dir, output_dir, all_files):
    """
    Renders a 3D volumetric plot of a random sample using PyVista.
    
    Args:
        data_dir (str): Directory containing the .npz files.
        output_dir (str): Directory to save the resulting plot.
        all_files (list): List of all .npz file names.
    """
    print("\n--- Generating 3D PyVista Plot ---")
    try:
        import pyvista as pv
    except ImportError:
        print("Error: 'pyvista' is not installed. Skipping 3D plot.")
        return

    file_to_plot = random.choice(all_files)
    file_path = os.path.join(data_dir, file_to_plot)
    data_3d = load_files(file_path)

    nz, ny, nx = data_3d.shape

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)

    grid.point_data['Facies'] = data_3d.transpose(2, 1, 0).flatten(order='F')

    custom_cmap, _ = create_colormap_and_legend()

    plotter = pv.Plotter(off_screen=True)
    plotter.add_volume(grid, scalars='Facies', cmap=custom_cmap, clim=[0, 13])

    plot_path = os.path.join(output_dir, f"3d_plot_{os.path.splitext(file_to_plot)[0]}.png")
    plotter.show(screenshot=plot_path)
    print(f"Saved 3D plot to: {plot_path}")


def main():
    """
    Main execution function. Parses arguments, applies plot styling,
    and coordinates the execution of enabled plotting functions.
    """
    parser = argparse.ArgumentParser(description="Generate all training sample visualizations.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the .npz dataset directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save plots (default: data_dir/plots)')
    parser.add_argument('--num_files', type=int, default=4, help='Number of files to sample for slice and entropy plots')
    parser.add_argument('--x_slice', type=int, default=None, help='X-index for orthogonal slice plot. Defaults to middle.')
    parser.add_argument('--y_slice', type=int, default=None, help='Y-index for orthogonal slice plot. Defaults to middle.')
    parser.add_argument('--z_slice', type=int, default=None, help='Z-index for orthogonal slice plot. Defaults to middle.')
    
    parser.add_argument('--disable_slices', action='store_true', help='Turn off orthogonal slice grid plots')
    parser.add_argument('--disable_dist', action='store_true', help='Turn off facies distribution analysis')    
    parser.add_argument('--disable_entropy', action='store_true', help='Turn off Entropy plots')
    parser.add_argument('--disable_3d', action='store_true', help='Turn off 3D PyVista plot')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(data_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {data_dir}")
    print(f"Saving figures to: {output_dir}")

    all_files = [f for f in os.listdir(data_dir) if f.endswith(('.npz', '.npy'))]
    if not all_files:
        print(f"Error: No .npz or .npy files found in {data_dir}.")
        return

    # Apply custom style configuration to Matplotlib
    apply_custom_plotting_flavor()

    if not args.disable_slices:
        plot_realization_slices(data_dir, output_dir, all_files, args.num_files, args.x_slice, args.y_slice, args.z_slice)

    if not args.disable_dist:
        plot_facies_distribution(data_dir, output_dir, all_files)

    if not args.disable_entropy:
        plot_entropy(data_dir, output_dir, all_files, args.num_files)
        
    if not args.disable_3d:
        plot_3d_pyvista(data_dir, output_dir, all_files)

    print("\n✅ All plots successfully generated.")

if __name__ == "__main__":
    main()

# EXAMPLE RUN: python scripts\Plots\training_dataset_summary.py --data_dir data\test_outputs --output_dir data\test_outputs_{add} --num_files 10