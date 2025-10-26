import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def remove_monopole(maps, verbose=True):
    """
    Remove the monopole (average value) from a set of maps.
    
    This function computes the mean value of each map and subtracts it from 
    all pixels in that map, effectively removing the monopole component.
    
    Args:
        maps (numpy.ndarray): Array of maps with shape (n_maps, height, width)
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
    
    Returns:
        numpy.ndarray: Maps with monopole removed, same shape as input
    """
    if verbose:  
        print('removing monopoles')

    # Compute the mean of each map across spatial dimensions
    maps_mean = np.mean(maps, axis=(1,2), dtype=np.float64)

    # Loop over all maps and subtract the mean value from each pixel
    for i in range(maps.shape[0]):
          maps[i] = maps[i] - maps_mean[i]

    return maps


class make_dataset2(Dataset):
    """
    PyTorch Dataset class for cosmology maps with on-the-fly data augmentation.
    
    This class creates a dataset from cosmology simulation maps and their parameters.
    Data augmentation (rotations and flips) is performed on-the-fly during data loading
    to save memory compared to pre-computing all augmented versions.
    
    The dataset handles:
    - Loading simulation parameters and maps
    - Normalizing parameters to [0,1] range
    - Log-scaling and normalizing map data
    - Optional monopole removal
    - Train/validation/test splits
    - On-the-fly random rotations (0°, 90°, 180°, 270°) and horizontal flips
    """
    
    def __init__(self, mode, seed, fmaps, fparams, splits, fmaps_norm, 
                 monopole, monopole_norm, verbose):
        """
        Initialize the dataset.
        
        Args:
            mode (str): Dataset mode - 'train', 'valid', 'test', or 'all'
            seed (int): Random seed for reproducible data splits
            fmaps (list): List of file paths to map data files (.npy format)
            fparams (str): Path to simulation parameters file
            splits (int): Number of maps per simulation to use
            fmaps_norm (list): List of normalization reference files (None for self-normalization)
            monopole (bool): Whether to remove monopole from maps
            monopole_norm (bool): Whether to remove monopole from normalization reference
            verbose (bool): Whether to print detailed progress information
        """
        super().__init__()

        # Load simulation parameters from file
        # Each row contains parameters for one simulation
        params_sims = np.loadtxt(fparams)
        total_sims, total_maps, num_params = params_sims.shape[0], params_sims.shape[0]*splits, params_sims.shape[1]

        # Initialize array to store parameters for each individual map
        # Since each simulation generates multiple maps, we replicate parameters
        params_maps = np.zeros((total_maps, num_params), dtype=np.float32)

        # Replicate simulation parameters for each map from that simulation
        for i in range(total_sims):
            for j in range(splits):
                params_maps[i*splits + j] = params_sims[i]

        # Normalize cosmological & astrophysical parameters using min-max scaling
        # Parameters: [Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2]
        # These ranges are specific to the IllustrisTNG simulation suite
        minimum     = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum     = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params_maps = (params_maps - minimum)/(maximum - minimum)

        # Determine data split based on mode
        # Standard 90%/5%/5% train/validation/test split
        if   mode=='train':  offset, size_sims = int(0.00*total_sims), int(0.90*total_sims)
        elif mode=='valid':  offset, size_sims = int(0.90*total_sims), int(0.05*total_sims)
        elif mode=='test':   offset, size_sims = int(0.95*total_sims), int(0.05*total_sims)
        elif mode=='all':    offset, size_sims = int(0.00*total_sims), int(1.00*total_sims)
        else:                raise Exception('Wrong name!')

        # Calculate total number of maps for this split
        size_maps = size_sims*splits

        # Create reproducible random shuffle of simulations
        # Important: we shuffle simulations, not individual maps, to avoid data leakage
        np.random.seed(seed)
        sim_numbers = np.arange(total_sims)
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims]
    
        # Generate map indices corresponding to selected simulations
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # Select parameters for chosen maps
        params_maps = params_maps[indexes]

        # Load and process map data
        channels = len(fmaps)  # Number of field channels (e.g., gas density, temperature)

        # Get map dimensions from first file
        dumb = np.load(fmaps[0])    
        height, width = dumb.shape[1], dumb.shape[2]
        del dumb

        # Initialize data array: (n_maps, n_channels, height, width)
        data = np.zeros((size_maps, channels, height, width), dtype=np.float32)

        print('Found %d channels\nReading data...'%channels)

        # Process each channel (field type) separately
        for channel, (fim, fnorm) in enumerate(zip(fmaps, fmaps_norm)):

            # Load raw map data for this channel
            data_c = np.load(fim)
            if data_c.shape[0]!=total_maps:
                raise Exception('sizes do not match')
            if verbose:
                print('%.3e < F(all|original) < %.3e'%(np.min(data_c), np.max(data_c)))

            # Apply log-scaling to handle large dynamic range
            # Replace zeros with ones to avoid log(0), then take signed log
            data_c = np.where(data_c !=0, data_c, 1)
            data_c = np.sign(data_c)*np.log10(np.abs(data_c))

            if verbose:
                print('%.3f < F(all|rescaled)  < %.3f'%(np.min(data_c), np.max(data_c)))

            # Optionally remove monopole (mean value) from each map
            if monopole is False:
                data_c = remove_monopole(data_c, verbose)

            # Normalize data using either self-statistics or external reference
            if fnorm is None:  
                # Self-normalization: use statistics from current dataset
                mean, std = np.mean(data_c), np.std(data_c)
            else:
                # External normalization: use statistics from reference dataset
                # This is useful for domain transfer (e.g., train on TNG, test on SIMBA)
                data_norm = np.load(fnorm)

                # Apply same preprocessing to normalization reference
                data_norm = np.where(data_norm !=0, data_norm, 1)
                data_norm = np.sign(data_norm)*np.log10(np.abs(data_norm))

                if monopole_norm is False:
                    data_norm = remove_monopole(data_norm, verbose)

                # Compute normalization statistics from reference data
                mean, std = np.mean(data_norm), np.std(data_norm)
                minimum, maximum = np.min(data_norm),  np.max(data_norm)

                del data_norm  # Free memory

            # Apply z-score normalization
            data_c = (data_c - mean)/std
            if verbose:
                print('%.3f < F(all|normalized) < %.3f'%(np.min(data_c), np.max(data_c))) 

            # Store processed data for selected maps
            data[:,channel,:,:] = data_c[indexes]

            if verbose:
                print('Channel %d contains %d maps'%(channel,size_maps))
                print('%.3f < F < %.3f'%(np.min(data_c), np.max(data_c)))

        # Convert to PyTorch tensors and store
        self.size = data.shape[0]
        self.x    = torch.from_numpy(data)
        self.y    = torch.from_numpy(params_maps)
        
        # Clean up memory
        del data, data_c, params_maps, params_sims

        print('{} dataset created!\n'.format(mode))
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset with random augmentation.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (augmented_maps, parameters) where:
                - augmented_maps: Tensor of shape (channels, height, width) with random rotation/flip
                - parameters: Tensor of normalized cosmological/astrophysical parameters
        """
        # Generate random augmentation: rotation (0°, 90°, 180°, 270°) and horizontal flip
        rot  = np.random.randint(0,4)  # 0-3 corresponding to 0°, 90°, 180°, 270°
        flip = np.random.randint(0,2)  # 0 or 1 for no flip or flip

        # Apply rotation (90-degree increments)
        maps = torch.rot90(self.x[idx], k=rot, dims=[1,2])
        
        # Apply horizontal flip with 50% probability
        if flip==1:  
            maps = torch.flip(maps, dims=[1])

        return maps, self.y[idx]


class make_dataset(Dataset):
    """
    PyTorch Dataset class for cosmology maps with pre-computed data augmentation.
    
    This class creates a dataset where all rotations and flips are pre-computed
    and stored in memory. This increases memory usage by 8x but eliminates
    augmentation overhead during training.
    
    Each original map generates 8 augmented versions:
    - 4 rotations (0°, 90°, 180°, 270°) × 2 flip states (original, horizontally flipped)
    """
    
    def __init__(self, mode, seed, fmaps, fparams, splits, fmaps_norm, 
                 monopole, monopole_norm, just_monopole, verbose):
        """
        Initialize the dataset with pre-computed augmentations.
        
        Args:
            mode (str): Dataset mode - 'train', 'valid', 'test', or 'all'
            seed (int): Random seed for reproducible data splits
            fmaps (list): List of file paths to map data files (.npy format)
            fparams (str): Path to simulation parameters file
            splits (int): Number of maps per simulation to use
            fmaps_norm (list): List of normalization reference files (None for self-normalization)
            monopole (bool): Whether to remove monopole from maps
            monopole_norm (bool): Whether to remove monopole from normalization reference
            just_monopole (bool): Whether to create monopole-only maps (constant value per map)
            verbose (bool): Whether to print detailed progress information
        """
        super().__init__()

        # Load and process simulation parameters (same as make_dataset2)
        params_sims = np.loadtxt(fparams)
        total_sims, total_maps, num_params = params_sims.shape[0], params_sims.shape[0]*splits, params_sims.shape[1]

        # Initialize array for map parameters
        params_maps = np.zeros((total_maps, num_params), dtype=np.float32)

        # Replicate simulation parameters for each map
        for i in range(total_sims):
            for j in range(splits):
                params_maps[i*splits + j] = params_sims[i]

        # Normalize parameters to [0,1] range
        minimum     = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum     = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params_maps = (params_maps - minimum)/(maximum - minimum)

        # Determine data split
        if   mode=='train':  offset, size_sims = int(0.00*total_sims), int(0.90*total_sims)
        elif mode=='valid':  offset, size_sims = int(0.90*total_sims), int(0.05*total_sims)
        elif mode=='test':   offset, size_sims = int(0.95*total_sims), int(0.05*total_sims)
        elif mode=='all':    offset, size_sims = int(0.00*total_sims), int(1.00*total_sims)
        else:                raise Exception('Wrong name!')

        size_maps = size_sims*splits

        # Create reproducible random shuffle
        np.random.seed(seed)
        sim_numbers = np.arange(total_sims)
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims]
    
        # Generate map indices for selected simulations
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # Select parameters for chosen maps
        params_maps = params_maps[indexes]

        # Initialize arrays for augmented data (8x larger due to augmentations)
        channels = len(fmaps)
        dumb = np.load(fmaps[0])    
        height, width = dumb.shape[1], dumb.shape[2]
        del dumb

        # Arrays to store all augmented versions
        data     = np.zeros((size_maps*8, channels, height, width), dtype=np.float32)
        params   = np.zeros((size_maps*8, num_params),              dtype=np.float32)

        print('Found %d channels\nReading data...'%channels)

        # Process each channel
        for channel, (fim, fnorm) in enumerate(zip(fmaps, fmaps_norm)):

            # Load and preprocess data (same as make_dataset2)
            data_c = np.load(fim)
            if data_c.shape[0]!=total_maps:
                raise Exception('sizes do not match')
            if verbose:
                print('%.3e < F(all|original) < %.3e'%(np.min(data_c), np.max(data_c)))

            # Log-scale transformation
            data_c = np.where(data_c !=0, data_c, 1)
            data_c = np.sign(data_c)*np.log10(np.abs(data_c))

            if verbose:
                print('%.3f < F(all|rescaled)  < %.3f'%(np.min(data_c), np.max(data_c)))

            # Remove monopole if requested
            if monopole is False:
                data_c = remove_monopole(data_c, verbose)

            # Handle normalization
            if fnorm is None:  
                mean, std = np.mean(data_c), np.std(data_c)
                minimum, maximum = np.min(data_c),  np.max(data_c)
            else:
                # External normalization reference
                data_norm = np.load(fnorm)
                data_norm = np.where(data_norm !=0, data_norm, 1)
                data_norm = np.sign(data_norm)*np.log10(np.abs(data_norm))

                if monopole_norm is False:
                    data_norm = remove_monopole(data_norm, verbose)

                mean, std = np.mean(data_norm), np.std(data_norm)
                minimum, maximum = np.min(data_norm),  np.max(data_norm)
                del data_norm

                # Special case: create monopole-only maps (for ablation studies)
                if just_monopole:
                    # Convert back to linear scale, compute mean, make constant maps
                    data_c = 10**(data_c)
                    mean_each_map = np.mean(data_c, axis=(1,2))
                    for i in range(data_c.shape[0]):
                        data_c[i] = mean_each_map[i]  # Set all pixels to mean value
                    data_c = np.log10(data_c)  # Convert back to log scale

            # Apply normalization
            data_c = (data_c - mean)/std
            if verbose:
                print('%.3f < F(all|normalized) < %.3f'%(np.min(data_c), np.max(data_c))) 

            # Select maps for this split
            data_c = data_c[indexes]

            # Generate all 8 augmented versions for each map
            counted_maps = 0
            
            # Loop over 4 rotation angles (0°, 90°, 180°, 270°)
            for rot in [0,1,2,3]:
                # Apply rotation
                data_rot = np.rot90(data_c, k=rot, axes=(1,2))

                # Store original rotation
                data[counted_maps:counted_maps+size_maps,channel,:,:] = data_rot
                params[counted_maps:counted_maps+size_maps]           = params_maps
                counted_maps += size_maps

                # Store horizontally flipped version
                data[counted_maps:counted_maps+size_maps,channel,:,:] = np.flip(data_rot, axis=1)
                params[counted_maps:counted_maps+size_maps]           = params_maps
                counted_maps += size_maps

                del data_rot

            if verbose:
                print('Channel %d contains %d maps'%(channel,counted_maps))
                print('%.3f < F < %.3f'%(np.min(data_c), np.max(data_c)))

        # Clean up intermediate variables
        del data_c, params_maps, params_sims

        # Convert to PyTorch tensors
        self.size = data.shape[0]
        self.x    = torch.from_numpy(data)
        self.y    = torch.from_numpy(params)

        del data, params  # Free numpy arrays

        print('{} dataset created!\n'.format(mode))
        
    def __len__(self):
        """Return the number of samples in the dataset (including all augmentations)."""
        return self.size

    def __getitem__(self, idx):
        """
        Get a sample from the pre-augmented dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (maps, parameters) where maps are already augmented
        """
        return self.x[idx], self.y[idx]





def create_dataloader(mode, seed, fmaps, fparams, batch_size, splits, 
                      fmaps_norm, monopole=True, monopole_norm=True,
                      rot_flip_in_mem=True, shuffle=True, 
                      just_monopole=False, verbose=False):
    """
    Create a PyTorch DataLoader for cosmology simulation data.
    
    This function creates either a memory-efficient dataset with on-the-fly augmentation
    or a pre-augmented dataset stored in memory, depending on the rot_flip_in_mem parameter.
    
    Args:
        mode (str): Dataset split - 'train', 'valid', 'test', or 'all'
        seed (int): Random seed for reproducible data splits
        fmaps (list): List of paths to map data files (.npy format)
        fparams (str): Path to simulation parameters file
        batch_size (int): Number of samples per batch
        splits (int): Number of maps per simulation to include
        fmaps_norm (list): List of normalization reference files (None for self-normalization)
        monopole (bool, optional): Whether to remove monopole from maps. Defaults to True.
        monopole_norm (bool, optional): Whether to remove monopole from norm reference. Defaults to True.
        rot_flip_in_mem (bool, optional): If True, pre-compute all augmentations in memory.
                                         If False, apply augmentations on-the-fly. Defaults to True.
        shuffle (bool, optional): Whether to shuffle data in DataLoader. Defaults to True.
        just_monopole (bool, optional): Whether to create monopole-only maps. Defaults to False.
        verbose (bool, optional): Whether to print detailed progress. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader for the dataset
        
    Note:
        - rot_flip_in_mem=True: Uses 8x more memory but faster training
        - rot_flip_in_mem=False: Memory efficient but slower due to on-the-fly augmentation
        - just_monopole=True: Creates maps with constant pixel values (for ablation studies)
    """
    # Choose dataset class based on memory/speed trade-off
    if rot_flip_in_mem:
        # Pre-compute all augmentations (faster training, more memory)
        data_set = make_dataset(mode, seed, fmaps, fparams, splits, 
                                fmaps_norm, monopole, monopole_norm, 
                                just_monopole, verbose)
    else:
        # On-the-fly augmentation (slower training, less memory)
        data_set = make_dataset2(mode, seed, fmaps, fparams, splits, 
                                fmaps_norm, monopole, monopole_norm, 
                                verbose)

    # Create DataLoader with specified batch size and shuffling
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)

    return data_loader


