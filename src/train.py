"""
Cosmological Parameter Estimation: CNN Training Pipeline
========================================================

This comprehensive training pipeline implements a deep learning approach for estimating
cosmological and astrophysical parameters from 2D simulation maps using convolutional
neural networks (CNNs). The pipeline is designed for the CAMELS project data.

Key Features:
- Hyperparameter optimization using Optuna
- Data augmentation and preprocessing
- Uncertainty quantification
- GPU acceleration support
- Comprehensive logging and model checkpointing
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# Import custom classes from other notebooks
from dataset import *
from network import *
from utils import *

# Core scientific computing and ML libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import sys
import os
import optuna
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Specialized libraries for cosmological data processing
import Pk_library as PKL
import density_field_library as DFL
import smoothing_library as SL


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

def setup_data_paths():
    """
    Configure data paths and parameters for CAMELS IllustrisTNG simulation data.
    
    Returns:
        dict: Dictionary containing all data configuration parameters
    """
    config = {
        # File paths for CAMELS data (using ~ as repo home)
        'fparams': '~/data/camels/params_IllustrisTNG.txt',
        'fmaps2': '~/data/camels/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy',
        'fmaps_norm': [None],  # No external normalization reference
        
        # Data processing parameters
        'seed': 1,        # Random seed for reproducible splits
        'channels': 1,    # Single field (total matter density)
        'splits': 15,     # Use all 15 maps per simulation
        
        # Experiment configuration
        'case': 'original',  # Options: 'original', 'min', 'max'
        'dens_cut_str': None  # Density threshold for augmentation (if applicable)
    }
    
    # Expand all paths
    config['fparams'] = expand_path(config['fparams'])
    config['fmaps2'] = expand_path(config['fmaps2'])
    
    return config

def augment_maps(experiment='density', case='min', dens_cut_str='1e11', kmax=None):
    """
    Apply data augmentation to cosmological maps.
    
    This function performs two types of augmentations:
    1. Density-based: Replace pixels above/below threshold with random values
    2. k-space filtering: Apply smoothing filters in Fourier space
    
    Args:
        experiment (str): Type of augmentation ('density' or 'kmax')
        case (str): Augmentation strategy ('min', 'max', or 'original')
        dens_cut_str (str): Density threshold as string (e.g., '1e11')
        kmax (float): Maximum k-mode for filtering (used in 'kmax' experiment)
    
    Note:
        - 'min' case: Replace low-density pixels with random values
        - 'max' case: Replace high-density pixels with random values
        - Augmented maps are saved to ~/results/augmented_maps/ directory
    """
    config = setup_data_paths()
    maps = np.load(config['fmaps2'])

    if experiment == 'density':
        dens_cut = float(dens_cut_str)
        
        if case == 'min':
            # Replace low-density pixels with random values in physical range
            indexes = np.where(maps < dens_cut)
            maps[indexes] = 10**(np.random.uniform(
                np.log10(np.nanmin(maps)), 
                np.log10(np.nanmax(maps)), 
                len(indexes[0])
            ))
            
        elif case == 'max':
            # Replace high-density pixels with random values
            indexes = np.where(maps > dens_cut)
            maps[indexes] = 10**(np.random.uniform(
                np.log10(np.nanmin(maps)), 
                np.log10(np.nanmax(maps)), 
                len(indexes[0])
            ))
            
        # Save augmented maps to repo results directory
        output_dir = expand_path(f'~/results/augmented_maps/{case}_density/')
        os.makedirs(output_dir, exist_ok=True)
        fmaps_save = f'{output_dir}/maps_Mtot_TNG_final_{dens_cut_str}.npy'
        with open(fmaps_save, 'wb') as f:
            np.save(f, maps)

    elif experiment == 'kmax':
        # Apply k-space filtering for frequency-based augmentation
        for i in range(maps.shape[0]):
            field = maps[i]
            BoxSize = 25.0        # Mpc/h - simulation box size
            R = 0.0               # Smoothing radius
            grid = field.shape[0]
            Filter = 'Top-Hat-k'  # Filter type in k-space
            kmin = 0              # Minimum k-mode
            threads = 28          # Parallel processing threads
            
            # Compute Fourier transform of the filter
            W_k = SL.FT_filter_2D(BoxSize, R, grid, Filter, threads, kmin, kmax)
            
            # Apply smoothing in Fourier space
            maps[i] = SL.field_smoothing_2D(field, W_k, threads)
            del field, W_k

        # Save filtered maps to repo results directory
        output_dir = expand_path('~/results/augmented_maps/kmax/')
        os.makedirs(output_dir, exist_ok=True)
        fmaps_save = f'{output_dir}/maps_Mtot_TNG_final_kmax_{kmax}.npy'
        with open(fmaps_save, 'wb') as f:
            np.save(f, maps)

def setup_device():
    """
    Configure PyTorch device (GPU/CPU) and optimization settings.
    
    Returns:
        torch.device: Configured device for model training
    """
    # Enable CUDA if available
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
        cudnn.benchmark = True  # Optimize for consistent input sizes
        
        # Report GPU information
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs Available")
        print(f'GPU model: {torch.cuda.get_device_name()}')
        
    else:
        print('CUDA Not Available - Using CPU')
        device = torch.device('cpu')
    
    return device

def visualize_sample_map(fmaps):
    """
    Display a sample cosmological map for visual inspection.
    
    Args:
        fmaps (list): List of file paths to map data
    """
    maps = np.load(fmaps[0])
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    plt.imshow(maps[0], norm=mcolors.LogNorm(), cmap='Greys')
    plt.colorbar(label='Total Matter Density')
    plt.title('Sample Cosmological Map (Log Scale)')
    plt.axis('off')
    plt.show()

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def fit(params, epochs, model, train_dl, valid_dl, trial, device, g, h, dir_wt):
    """
    Train the CNN model for cosmological parameter estimation.
    
    This function implements the complete training loop with:
    - Custom loss function for uncertainty quantification
    - Learning rate scheduling
    - Model checkpointing based on validation performance
    - Comprehensive logging
    
    Args:
        params (dict): Hyperparameters including learning rate and weight decay
        epochs (int): Number of training epochs
        model (nn.Module): CNN model to train
        train_dl (DataLoader): Training data loader
        valid_dl (DataLoader): Validation data loader
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization
        device (torch.device): Computing device (GPU/CPU)
        g (list): Indices for parameter predictions
        h (list): Indices for uncertainty estimates
        dir_wt (str): Directory to save model weights and logs
    
    Returns:
        float: Best validation loss achieved during training
        
    Loss Function:
        The loss combines two components:
        1. Mean squared error between predictions and targets
        2. Uncertainty calibration term ensuring error estimates match actual errors
        
        Loss = log(MSE) + log(Uncertainty_Calibration_Error)
    """
    min_valid_loss = 1e34  # Initialize with very large value
    
    # Extract hyperparameters
    max_lr = params['max_lr']
    wd = params["wd"]
    beta1, beta2 = 0.5, 0.999  # Adam optimizer parameters
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=max_lr, 
        weight_decay=wd, 
        betas=(beta1, beta2)
    )
    
    # Cyclic learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-09, 
        max_lr=max_lr, 
        cycle_momentum=False, 
        step_size_up=500, 
        step_size_down=500
    )

    for epoch in range(epochs):
        # ====================================================================
        # TRAINING PHASE
        # ====================================================================
        train_loss1 = torch.zeros(len(g)).to(device)  # MSE loss per parameter
        train_loss2 = torch.zeros(len(g)).to(device)  # Uncertainty calibration loss
        train_loss, points = 0.0, 0
        
        model.train()  # Set model to training mode
        
        for x, y in train_dl:
            bs = x.shape[0]           # Batch size
            x = x.to(device)          # Input maps
            y = y.to(device)[:, g]    # Target parameters
            
            # Forward pass
            p = model(x)              # Model predictions
            y_NN = p[:, g]            # Predicted parameter values
            e_NN = p[:, h]            # Predicted uncertainties
            
            # Compute loss components
            # Loss 1: Mean squared error between predictions and targets
            loss1 = torch.mean((y_NN - y)**2, axis=0)
            
            # Loss 2: Uncertainty calibration - ensures predicted uncertainties
            # match actual prediction errors
            loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
            
            # Combined loss (log of both components)
            loss = torch.mean(torch.log(loss1) + torch.log(loss2))
            
            # Accumulate losses for reporting
            train_loss1 += loss1 * bs
            train_loss2 += loss2 * bs
            points += bs
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Compute average training loss
        train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
        train_loss = torch.mean(train_loss).item()

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        valid_loss1 = torch.zeros(len(g)).to(device)
        valid_loss2 = torch.zeros(len(g)).to(device)
        valid_loss, points = 0.0, 0
        
        model.eval()  # Set model to evaluation mode
        
        for x, y in valid_dl:
            with torch.no_grad():  # Disable gradient computation for efficiency
                bs = x.shape[0]
                x = x.to(device)
                y = y.to(device)[:, g]
                
                # Forward pass
                p = model(x)
                y_NN = p[:, g]
                e_NN = p[:, h]
                
                # Compute validation losses (same as training)
                loss1 = torch.mean((y_NN - y)**2, axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss = torch.mean(torch.log(loss1) + torch.log(loss2))
                
                valid_loss1 += loss1 * bs
                valid_loss2 += loss2 * bs
                points += bs
        
        # Compute average validation loss
        valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
        valid_loss = torch.mean(valid_loss).item()

        # ====================================================================
        # MODEL CHECKPOINTING AND LOGGING
        # ====================================================================
        
        # Save model if validation loss improved
        if valid_loss < min_valid_loss:
            fweights = f'{dir_wt}/weights_{trial.number}.pt'
            torch.save(model.state_dict(), fweights)
            min_valid_loss = valid_loss

        # Log training progress
        floss = f'{dir_wt}/losses_{trial.number}.txt'
        with open(floss, 'a') as f:
            f.write(f'{epoch} {train_loss:.5e} {valid_loss:.5e}\n')
        
    return min_valid_loss

def objective(trial, train_dl, valid_dl, device, g, h, dir_wt, channels):
    """
    Objective function for Optuna hyperparameter optimization.
    
    This function defines the hyperparameter search space and trains a model
    with the suggested hyperparameters, returning the validation loss for optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial object with hyperparameter suggestions
        train_dl (DataLoader): Training data loader
        valid_dl (DataLoader): Validation data loader
        device (torch.device): Computing device
        g (list): Parameter prediction indices
        h (list): Uncertainty prediction indices
        dir_wt (str): Directory for saving weights
        channels (int): Number of input channels
    
    Returns:
        float: Final validation loss for this hyperparameter configuration
        
    Hyperparameter Search Space:
        - max_lr: Learning rate (1e-5 to 5e-3, log scale)
        - wd: Weight decay (1e-8 to 1e-1, log scale)
        - dr: Dropout rate (0.0 to 0.9)
        - hidden: Base hidden units (6 to 12)
    """
    # Define hyperparameter search space
    params = {
        'max_lr': trial.suggest_float("max_lr", 1e-5, 5e-3, log=True),
        'wd': trial.suggest_float("wd", 1e-8, 1e-1, log=True),
        'dr': trial.suggest_float("dr", 0.0, 0.9),
        'hidden': trial.suggest_int("hidden", 6, 12)
    }
    
    # Initialize model with suggested hyperparameters
    model = Model_CMD(params, channels)
    
    # Enable multi-GPU training if available
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Train model for specified number of epochs
    epochs = 200
    loss_fin = fit(params, epochs, model, train_dl, valid_dl, trial, device, g, h, dir_wt)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return loss_fin

def run_training_pipeline():
    """
    Execute the complete training pipeline for cosmological parameter estimation.
    
    This is the main function that orchestrates:
    1. Data setup and preprocessing
    2. Model configuration
    3. Hyperparameter optimization using Optuna
    4. Training and validation
    5. Results logging and model saving
    
    The pipeline uses the CAMELS IllustrisTNG simulation data and implements
    a CNN-based approach for estimating cosmological and astrophysical parameters
    with uncertainty quantification.
    
    All outputs are saved relative to the repository home directory using ~ notation.
    """
    print("="*80)
    print("COSMOLOGICAL PARAMETER ESTIMATION: CNN TRAINING PIPELINE")
    print("="*80)
    print(f"Repository home: {get_repo_home()}")
    
    # ========================================================================
    # CONFIGURATION AND SETUP
    # ========================================================================
    
    config = setup_data_paths()
    device = setup_device()
    
    # Determine map file paths based on experiment type
    if config['case'] == 'original':
        fmaps = [config['fmaps2']]
        dir_wt = expand_path(f'~/results/models/density/{config["case"]}')
    else:
        fmaps = [expand_path(f'~/results/augmented_maps/{config["case"]}_density/maps_Mtot_TNG_final_{config["dens_cut_str"]}.npy')]
        dir_wt = expand_path(f'~/results/models/density/{config["case"]}/{config["dens_cut_str"]}')
    
    # Create output directory
    os.makedirs(dir_wt, exist_ok=True)
    print(f'Output directory: {dir_wt}')
    print(f'Maps location: {fmaps[0]}')
    
    # Check if data files exist
    if not os.path.exists(fmaps[0]):
        print(f"WARNING: Map file not found at {fmaps[0]}")
        print("Please ensure the data files are placed in the correct ~/data/ directory")
        return None, None
    
    if not os.path.exists(config['fparams']):
        print(f"WARNING: Parameters file not found at {config['fparams']}")
        print("Please ensure the data files are placed in the correct ~/data/ directory")
        return None, None
    
    # Visualize sample data
    visualize_sample_map(fmaps)
    
    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    print("\nPreparing datasets...")
    start_time = time.time()
    
    batch_size = 128
    
    # Create training DataLoader
    print('Preparing training set...')
    train_dl = create_dataloader(
        'train', config['seed'], fmaps, config['fparams'], 
        batch_size, config['splits'], config['fmaps_norm'], 
        rot_flip_in_mem=False, verbose=True
    )
    
    # Create validation DataLoader  
    print('Preparing validation set...')
    valid_dl = create_dataloader(
        'valid', config['seed'], fmaps, config['fparams'], 
        batch_size, config['splits'], config['fmaps_norm'], 
        rot_flip_in_mem=False, verbose=True
    )
    
    data_time = time.time() - start_time
    print(f'Dataset preparation time: {data_time/3600:.4f} hours')
    
    # ========================================================================
    # PARAMETER CONFIGURATION
    # ========================================================================
    
    # Define which parameters to predict
    # 0: Omega_m (matter density parameter)
    # 1: sigma_8 (amplitude of matter fluctuations)
    # 2: A_SN1 (supernova feedback 1)
    # 3: A_AGN1 (AGN feedback 1) 
    # 4: A_SN2 (supernova feedback 2)
    # 5: A_AGN2 (AGN feedback 2)
    params_to_predict = [0, 1, 2, 3, 4, 5]  # Predict all 6 parameters
    
    g = params_to_predict              # Indices for parameter means
    h = [6 + i for i in g]            # Indices for parameter uncertainties
    
    print(f"Predicting parameters: {params_to_predict}")
    print(f"Parameter indices: {g}")
    print(f"Uncertainty indices: {h}")
    
    # ========================================================================
    # HYPERPARAMETER OPTIMIZATION WITH OPTUNA
    # ========================================================================
    
    # Configure Optuna study with paths relative to repo home
    if config['case'] == 'original':
        study_name = f'Mtot_dens_{config["case"]}'
        storage = f'sqlite:///{dir_wt}/Mtot_dens_{config["case"]}.db'
    else:
        study_name = f'Mtot_dens_{config["case"]}_{config["dens_cut_str"]}'
        storage = f'sqlite:///{dir_wt}/Mtot_dens_{config["case"]}_{config["dens_cut_str"]}.db'
    
    print(f"\nStarting hyperparameter optimization...")
    print(f"Study name: {study_name}")
    print(f"Database: {storage}")
    
    start_time = time.time()
    
    # Create or load existing study
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(), 
        study_name=study_name, 
        storage=storage, 
        load_if_exists=True
    )
    
    # Run optimization trials
    study.optimize(
        lambda trial: objective(trial, train_dl, valid_dl, device, g, h, dir_wt, config['channels']), 
        n_trials=50
    )
    
    optimization_time = time.time() - start_time
    print(f'Hyperparameter optimization completed in {optimization_time/3600:.4f} hours')
    

    
    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================
    
    # Load best hyperparameters
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best trial number: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.6f}")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest model weights saved to: {dir_wt}/weights_{best_trial.number}.pt")
    print(f"Training logs saved to: {dir_wt}/losses_{best_trial.number}.txt")
    
    total_time = data_time + optimization_time
    print(f"\nTotal pipeline execution time: {total_time/3600:.4f} hours")
    
    # Create a summary file with results
    summary_path = f'{dir_wt}/training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("CNN Training Pipeline Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Repository home: {get_repo_home()}\n")
        f.write(f"Data source: {fmaps[0]}\n")
        f.write(f"Parameters file: {config['fparams']}\n")
        f.write(f"Output directory: {dir_wt}\n")
        f.write(f"Best trial: {best_trial.number}\n")
        f.write(f"Best loss: {best_trial.value:.6f}\n")
        f.write(f"Total time: {total_time/3600:.4f} hours\n")
        f.write("\nBest hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Training summary saved to: {summary_path}")
    
    return best_trial, dir_wt

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Execute the complete training pipeline
    best_trial, output_dir = run_training_pipeline()
    
    if best_trial is not None:
        print("\n🎉 Training pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Best model: trial_{best_trial.number} with loss {best_trial.value:.6f}")
    else:
        print("Training pipeline failed - please check data file paths")