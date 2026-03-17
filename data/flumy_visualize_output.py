import numpy as np
import pyvista as pv
import os

def visualize_npz(file_path):
    """
    Visualizes a 3D numpy array from a .npz file using PyVista.
    The data is assumed to be one-hot encoded, and will be converted
    to scalar values for visualization.

    Parameters:
    - file_path (str): The path to the .npz file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load the data from the .npz file
    try:
        with np.load(file_path) as data:
            # Check available arrays in the npz file
            if not data.files:
                print("Error: NPZ file is empty.")
                return
            
            # Assume the main data array is the first one
            arr_name = data.files[0]
            one_hot_data = data[arr_name]
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    print(f"Loaded array '{arr_name}' with shape: {one_hot_data.shape}")

    if one_hot_data.ndim != 4:
        print(f"Error: Expected a 4D one-hot encoded array (width, height, depth, classes), but got {one_hot_data.ndim}D.")
        return

    # Convert one-hot encoded data to scalar data
    # We assume the first axis is the one-hot encoded dimension based on the shape (3, 64, 128, 128)
    scalar_data = np.argmax(one_hot_data, axis=0)
    
    # Create a PyVista ImageData object (which is a uniform grid).
    grid = pv.ImageData()
    
    # Set the dimensions of the grid. PyVista expects (nx, ny, nz).
    # We assume the numpy array shape is (nx, ny, nz).
    grid.dimensions = scalar_data.shape

    # Add the scalar data to the grid.
    # The `ravel()` function flattens the array. 'F' order is important
    # to match PyVista's memory layout.
    grid.cell_data['scalars'] = scalar_data.ravel(order='F')

    # Create a plotter and add the grid to it
    plotter = pv.Plotter()
    
    # Add the grid as a volume. This is often better for this kind of data.
    # You can also use `plotter.add_mesh(grid, show_edges=True)` for a surface view
    plotter.add_volume(grid, cmap="viridis")
    
    # Show the plot
    print("Displaying 3D visualization. Close the window to exit.")
    plotter.show()

if __name__ == '__main__':
    # This script is intended to be run from the project root directory.
    file_to_visualize = os.path.join('data', 'test_outputs', 'sample_43.npz')

    if not os.path.exists(file_to_visualize):
        print(f"Error: File not found at '{file_to_visualize}'")
        print("Please ensure you are running this script from the project's root directory.")
    else:
        visualize_npz(file_to_visualize)
