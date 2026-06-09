# Graph Report - .  (2026-06-08)

## Corpus Check
- 57 files · ~303,628 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 360 nodes · 415 edges · 30 communities (29 shown, 1 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Facies Config|Facies Config]]
- [[_COMMUNITY_Post-Processing Realizations|Post-Processing Realizations]]
- [[_COMMUNITY_Training Dataset Summary|Training Dataset Summary]]
- [[_COMMUNITY_Old Flumy Utilities|Old Flumy Utilities]]
- [[_COMMUNITY_Inversion Loss Optimization|Inversion Loss Optimization]]
- [[_COMMUNITY_Facies Dataset Loader|Facies Dataset Loader]]
- [[_COMMUNITY_GAN Training Pipeline|GAN Training Pipeline]]
- [[_COMMUNITY_Facies Preprocessing Datasets|Facies Preprocessing Datasets]]
- [[_COMMUNITY_Flumy Worker Executable|Flumy Worker Executable]]
- [[_COMMUNITY_Post-Processing Spatial Entropy|Post-Processing Spatial Entropy]]
- [[_COMMUNITY_Preprocessing Custom Plots|Preprocessing Custom Plots]]
- [[_COMMUNITY_Postprocessing Custom Plots|Postprocessing Custom Plots]]
- [[_COMMUNITY_Core Custom Plots|Core Custom Plots]]
- [[_COMMUNITY_Scripts Custom Plots|Scripts Custom Plots]]
- [[_COMMUNITY_Entropy Slices Plots|Entropy Slices Plots]]
- [[_COMMUNITY_Flumy Nexus Run Pipeline|Flumy Nexus Run Pipeline]]
- [[_COMMUNITY_Training Config Utilities|Training Config Utilities]]
- [[_COMMUNITY_Core Config Utilities|Core Config Utilities]]
- [[_COMMUNITY_Conda Environment Configurations|Conda Environment Configurations]]
- [[_COMMUNITY_Data Preprocessing Crops|Data Preprocessing Crops]]
- [[_COMMUNITY_Old Flumy Batch Worker|Old Flumy Batch Worker]]
- [[_COMMUNITY_Old Flumy Parallel Worker|Old Flumy Parallel Worker]]
- [[_COMMUNITY_Training Visualizer Generator|Training Visualizer Generator]]
- [[_COMMUNITY_Flumy Conda Environment|Flumy Conda Environment]]
- [[_COMMUNITY_GAN Visualizer Generator|GAN Visualizer Generator]]
- [[_COMMUNITY_GAN Pipeline Configuration|GAN Pipeline Configuration]]
- [[_COMMUNITY_Project Readme Documentation|Project Readme Documentation]]
- [[_COMMUNITY_Flumy Visualization Output|Flumy Visualization Output]]

## God Nodes (most connected - your core abstractions)
1. `PostProcessing` - 14 edges
2. `FaciesDataset` - 11 edges
3. `Cell-Wise Entropy` - 8 edges
4. `FaciesDataset` - 7 edges
5. `FlumyDataManager` - 7 edges
6. `main()` - 7 edges
7. `run_flumy_worker()` - 6 edges
8. `FaciesDataset` - 6 edges
9. `MultiChannelContextLoss` - 6 edges
10. `LPIPSLoss` - 6 edges

## Surprising Connections (you probably didn't know these)
- `ZX Plane Cell-Wise Entropy Plot (100 Realizations)` --conceptually_related_to--> `Spatial Entropy Mapping`  [INFERRED]
  plots/post_editing_plots/RUN_10000_of_20000_samples_128xy_dataset_50_epochs_bs_64_val_size_020_double_conv/entropy_matrix_ZX_realizations_100_samples.png → scripts/gan_pipeline/04_postprocessing/postprocessing.py
- `XY Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing` --conceptually_related_to--> `Spatial Entropy Mapping`  [INFERRED]
  plots/post_training_plots/RUN_10000_samples_5_epochs_bs_64_val_size_015_one_hot_label_smoothing/entropy_matrix_XY_realizations_100_samples.png → scripts/gan_pipeline/04_postprocessing/postprocessing.py
- `ZX Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing` --conceptually_related_to--> `Spatial Entropy Mapping`  [INFERRED]
  plots/post_training_plots/RUN_10000_samples_5_epochs_bs_64_val_size_015_one_hot_label_smoothing/entropy_matrix_ZX_realizations_100_samples.png → scripts/gan_pipeline/04_postprocessing/postprocessing.py
- `ZY Plane Cell-Wise Entropy Plot (100 Realizations)` --conceptually_related_to--> `Spatial Entropy Mapping`  [INFERRED]
  plots/post_editing_plots/RUN_10000_of_20000_samples_128xy_dataset_50_epochs_bs_64_val_size_020_double_conv/entropy_matrix_ZY_realizations_100_samples.png → scripts/gan_pipeline/04_postprocessing/postprocessing.py
- `ZY Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing` --conceptually_related_to--> `Spatial Entropy Mapping`  [INFERRED]
  plots/post_training_plots/RUN_10000_samples_5_epochs_bs_64_val_size_015_one_hot_label_smoothing/entropy_matrix_ZY_realizations_100_samples.png → scripts/gan_pipeline/04_postprocessing/postprocessing.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Project Conda Environments** — environments_flumy_environment, environments_fluvgan_environment, environments_fluvgan_environment_hpc [INFERRED 0.85]
- **Hybrid CLI and JSON Configuration System** — scripts_gan_pipeline_02_training_readme_hybrid_config, scripts_old_scripts_old_gan_scripts_gan_readme_hybrid_config, scripts_gan_gan_training [INFERRED 0.85]

## Communities (30 total, 1 thin omitted)

### Community 0 - "Facies Config"
Cohesion: 0.05
Nodes (42): channel_lag, color, val, crevasse_channel, color, val, crevasse_splay_core, color (+34 more)

### Community 1 - "Post-Processing Realizations"
Cohesion: 0.10
Nodes (14): PostProcessing, Maps 0-indexed categorical data to specific geological facies codes., Creates a ListedColormap and legend patches based on facies_properties., Extracts patterns and their raw counts from a single 3D array.          Uses a, Finds connected 3D bodies of a specific facies and returns their volumes., Helper function to calculate and plot the entropy matrices with dynamic titling., Plots the continuous and discrete percentage distributions across all GAN realiz, Handles spatial and statistical validation metrics for GAN-generated geological (+6 more)

### Community 2 - "Training Dataset Summary"
Cohesion: 0.12
Nodes (24): load_array(), main(), parse_args(), This script converts a directory of .npz/.npy files containing 3D facies data in, Helper function to correctly load either .npz or .npy files., create_colormap_and_legend(), load_files(), main() (+16 more)

### Community 3 - "Old Flumy Utilities"
Cohesion: 0.10
Nodes (15): BatchSimulator, FlumyDataManager, Args:             output_dit (str): sets output directory for training, validat, Pre-allocates an HDF5 file with gzip compression for a specific batch., Writes a single 3D realization into a specific index slot of an existing HDF5 fi, Orchestrates the parallel execution of simulation tasks across multiple CPU core, Args:             worker_function (Callable): The function executed by each par, Manages the creation of HDF5 files for 3D geological data (+7 more)

### Community 4 - "Inversion Loss Optimization"
Cohesion: 0.13
Nodes (13): main(), map_facies(), MultiChannelContextLoss, ============================================== GAN inversion using latent-vector, Context loss for multi-channel one-hot encoded facies.     Expands a mask based, Maps raw facies values to 3 classes as defined in dataloader., LPIPSLoss, main() (+5 more)

### Community 5 - "Facies Dataset Loader"
Cohesion: 0.12
Nodes (8): FaciesDataset, Saves the facies mapping configuration to a JSON file.          Creates the di, FaciesDataset, A PyTorch Dataset for loading and processing facies data from an HDF5 file., Retrieves and processes a specific sample from the dataset.          Fetches t, Initializes the FaciesDataset.          Args:             h5_path (str): Path, Saves the facies mapping configuration to a JSON file.          Creates the di, Dataset

### Community 6 - "GAN Training Pipeline"
Cohesion: 0.12
Nodes (8): FaciesDataset, A PyTorch Dataset for loading and processing facies data from an HDF5 file., Retrieves and processes a specific sample from the dataset.         *DEBUG FEAT, Initializes the FaciesDataset.              Args:                 h5_path (st, Saves the facies mapping configuration to a JSON file.          Creates the di, Architecture DCGAN              + LeakyReLU in the generator              + Bina, Architecture DCGAN              + LeakyReLU in the generator              + Bina, The architecture of this model are based on Rongier & Peeters., 2025.  Archite

### Community 7 - "Facies Preprocessing Datasets"
Cohesion: 0.19
Nodes (7): Bar chart comparing facies percentages between two setting_1_nexus datasets. Point Bar facies dominates both at ~66-67%, followed by Mud Plug (~12-13%), Overbank (~11%), and Levee (~5-6%)., Facies Distribution, Mud Plug Facies, Overbank Facies, Point Bar Facies, Summary Setting 1 Nexus 1000 Samples NTG 67 Channel Depth 5 ISBX 80 Plot, Summary Setting 1 Nexus 1000 Samples NTG 67 Channel Depth 6 ISBX 100 Plot

### Community 8 - "Flumy Worker Executable"
Cohesion: 0.19
Nodes (11): main(), os_check(), Checks the operating system and sets the appropriate paths for the Flumy executa, groupFacies(), Saves the dictionary of arrays to the specified format., Groups facies produced by the flumy simulation.     Default scheme maps into FA, save_sample(), batch_writer() (+3 more)

### Community 9 - "Post-Processing Spatial Entropy"
Cohesion: 0.27
Nodes (10): The purpose of this file is to validate the realization produced by the GAN mode, Cell-Wise Entropy, XY Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing, ZX Plane Cell-Wise Entropy Plot (100 Realizations), ZX Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing, ZY Plane Cell-Wise Entropy Plot (100 Realizations), ZY Plane Cell-Wise Entropy Plot (100 Realizations) - Label Smoothing, Run Config: 10000 of 20000 samples, 128xy, 50 epochs, BS 64, Val 0.20, Double Conv (+2 more)

### Community 10 - "Preprocessing Custom Plots"
Cohesion: 0.18
Nodes (9): apply_custom_plotting_flavor(), FaciesColorMap, main(), parse_args(), This script contains custom plotting configurations and colormaps used across th, Parses command-line arguments., Custom colormap for facies visualization based on the provided color scheme., Creates a BoundaryNorm mapped strictly to the discrete facies integers. (+1 more)

### Community 11 - "Postprocessing Custom Plots"
Cohesion: 0.18
Nodes (9): apply_custom_plotting_flavor(), FaciesColorMap, main(), parse_args(), This script contains custom plotting configurations and colormaps used across th, Parses command-line arguments., Custom colormap for facies visualization based on the provided color scheme., Creates a BoundaryNorm mapped strictly to the discrete facies integers. (+1 more)

### Community 12 - "Core Custom Plots"
Cohesion: 0.18
Nodes (9): apply_custom_plotting_flavor(), FaciesColorMap, main(), parse_args(), This script contains custom plotting configurations and colormaps used across th, Parses command-line arguments., Custom colormap for facies visualization based on the provided color scheme., Creates a BoundaryNorm mapped strictly to the discrete facies integers. (+1 more)

### Community 13 - "Scripts Custom Plots"
Cohesion: 0.18
Nodes (9): apply_custom_plotting_flavor(), FaciesColorMap, main(), parse_args(), This script contains custom plotting configurations and colormaps used across th, Parses command-line arguments., Custom colormap for facies visualization based on the provided color scheme., Creates a BoundaryNorm mapped strictly to the discrete facies integers. (+1 more)

### Community 14 - "Entropy Slices Plots"
Cohesion: 0.31
Nodes (11): Cell-Wise Entropy, Entropy Matrix XY Realizations (1000 Samples) Plot, Entropy Matrix XY Realizations (100 Samples) Plot, Entropy Matrix ZX Realizations (1000 Samples) Plot, Entropy Matrix ZX Realizations (100 Samples) Plot, Entropy Matrix ZY Realizations (1000 Samples) Plot, Entropy Matrix ZY Realizations (100 Samples) Plot, Entropy Matrix XY Realizations (100 Samples) Plot (Post-Optimization) (+3 more)

### Community 15 - "Flumy Nexus Run Pipeline"
Cohesion: 0.25
Nodes (10): flumy_worker(), main(), os_check(), parse_args(), Flumy runner script utilizing the native Python API for precise Net-to-Gross con, Checks the operating system and sets the appropriate base data directory.     R, Creates the necessary directory structure for saving samples., Worker function to generate a single Flumy simulation with automatic retries. (+2 more)

### Community 16 - "Training Config Utilities"
Cohesion: 0.25
Nodes (8): load_config(), parse_hybrid_args(), Saves a configuration dictionary to a JSON file.          Args:         confi, Parses arguments using a hybrid approach: CLI overrides JSON configs., Loads a JSON configuration file.          Args:         config_path (str): Pa, Validates the HDF5 dataset before training to catch corrupted or malformed files, save_config(), validate_dataset()

### Community 17 - "Core Config Utilities"
Cohesion: 0.25
Nodes (8): load_config(), parse_hybrid_args(), Saves a configuration dictionary to a JSON file.          Args:         confi, Parses arguments using a hybrid approach: CLI overrides JSON configs., Loads a JSON configuration file.          Args:         config_path (str): Pa, Validates the HDF5 dataset before training to catch corrupted or malformed files, save_config(), validate_dataset()

### Community 18 - "Conda Environment Configurations"
Cohesion: 0.22
Nodes (9): FluvGAN Environment Configuration, FluvGAN HPC Environment Configuration, Flumy Package (HPC), Pyro PPL Package (HPC), PyVista Package (HPC), VoxGAN Package (HPC), Pyro PPL Package, PyVista Package (FluvGAN) (+1 more)

### Community 19 - "Data Preprocessing Crops"
Cohesion: 0.36
Nodes (7): center_crop(), load_array(), main(), parse_args(), This script converts a directory of .npz/.npy files containing 3D facies data in, Helper function to correctly load either .npz or .npy files., Crops the specified axis to target_dim by taking the middle slices.

### Community 20 - "Old Flumy Batch Worker"
Cohesion: 0.47
Nodes (5): groupFacies(), one_hot_encode_3d(), Converts a 3D categorical array into a 4D one-hot encoded array.          Args, Function to group all facies produced by the flumy simulation into three main ca, worker_flumy_8501()

### Community 21 - "Old Flumy Parallel Worker"
Cohesion: 0.47
Nodes (5): groupFacies(), one_hot_encode_3d(), Function to group all facies produced by the flumy simulation into three main ca, Converts a 3D categorical array into a 4D one-hot encoded array.          Args, worker_flumy_8501()

### Community 22 - "Training Visualizer Generator"
Cohesion: 0.60
Nodes (3): generate_realizations(), main(), parse_args()

### Community 23 - "Flumy Conda Environment"
Cohesion: 0.40
Nodes (5): Flumy Environment Configuration, Flumy Python Package, ITKWidgets Package, PyVista Package, Trame Package

### Community 24 - "GAN Visualizer Generator"
Cohesion: 0.60
Nodes (3): generate_realizations(), main(), parse_args()

### Community 25 - "GAN Pipeline Configuration"
Cohesion: 0.80
Nodes (5): GAN Training Script, GAN Pipeline Training README, Hybrid Configuration System, Old GAN README, Hybrid Configuration System (Old)

### Community 26 - "Project Readme Documentation"
Cohesion: 0.67
Nodes (3): Data Generation README, Project README, Deep Generative Models for 3D Geological Reservoir Modeling

## Knowledge Gaps
- **42 isolated node(s):** `val`, `color`, `val`, `color`, `val` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PostProcessing` connect `Post-Processing Realizations` to `Post-Processing Spatial Entropy`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `FaciesDataset` connect `GAN Training Pipeline` to `Facies Dataset Loader`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Why does `plot_facies_distribution()` connect `Training Dataset Summary` to `Data Preprocessing Crops`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **What connects `Checks the operating system and sets the appropriate paths for the Flumy executa`, `Groups facies produced by the flumy simulation.     Default scheme maps into FA`, `Saves the dictionary of arrays to the specified format.` to the rest of the system?**
  _140 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Facies Config` be split into smaller, more focused modules?**
  _Cohesion score 0.046511627906976744 - nodes in this community are weakly interconnected._
- **Should `Post-Processing Realizations` be split into smaller, more focused modules?**
  _Cohesion score 0.10461538461538461 - nodes in this community are weakly interconnected._
- **Should `Training Dataset Summary` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._