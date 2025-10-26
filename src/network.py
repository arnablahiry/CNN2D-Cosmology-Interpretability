import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model_CMD(nn.Module):
    """
    Convolutional Neural Network for Cosmological Parameter Estimation.
    
    This CNN model is based on the O3 architecture from Villaescusa-Navarro et al.
    It processes 2D cosmological simulation maps (e.g., gas density, temperature) 
    and predicts cosmological and astrophysical parameters along with their uncertainties.
    
    Architecture Overview:
    - 6 convolutional blocks with progressive downsampling (256×256 → 1×1)
    - Each block contains: Conv → Conv → Conv/Downsample + BatchNorm + LeakyReLU
    - Circular padding to handle periodic boundary conditions in simulations
    - Fully connected layers for final parameter prediction
    - Outputs 12 values: 6 parameters + 6 uncertainties
    
    The model handles:
    - Multi-channel input (e.g., gas density + temperature)
    - Cosmological parameters: Ωₘ, σ₈
    - Astrophysical parameters: A_SN1, A_AGN1, A_SN2, A_AGN2
    - Uncertainty quantification for each parameter
    """
    
    def __init__(self, params, channels):
        """
        Initialize the CNN model.
        
        Args:
            params (dict): Dictionary containing model hyperparameters
                - "hidden" (int): Base number of hidden units (multiplied by 2, 4, 8, etc.)
                - "dr" (float): Dropout rate for regularization
            channels (int): Number of input channels (e.g., 1 for density only, 2 for density+temperature)
            
        Note:
            The architecture progressively increases channel depth:
            Input channels → 2×hidden → 4×hidden → 8×hidden → 16×hidden → 32×hidden → 64×hidden → 128×hidden
        """
        super(Model_CMD, self).__init__()

        hidden = params["hidden"]  # Base number of hidden units
        dr = params["dr"]          # Dropout rate

        # ============================================================================
        # BLOCK 0: Input processing (256×256 → 128×128)
        # ============================================================================
        # First block processes raw input and reduces spatial dimensions by 2
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)  # Feature extraction
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)  # Feature refinement
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)  # Downsampling
        
        # Batch normalization for stable training
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)

        # ============================================================================
        # BLOCK 1: Feature extraction (128×128 → 64×64)
        # ============================================================================
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # Channel expansion
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # Feature processing
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)  # Spatial reduction
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)

        # ============================================================================
        # BLOCK 2: Mid-level features (64×64 → 32×32)
        # ============================================================================
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # Deeper feature extraction
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)

        # ============================================================================
        # BLOCK 3: High-level features (32×32 → 16×16)
        # ============================================================================
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # Abstract feature learning
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)

        # ============================================================================
        # BLOCK 4: Abstract features (16×16 → 8×8)
        # ============================================================================
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # High-level pattern recognition
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)

        # ============================================================================
        # BLOCK 5: Deep abstract features (8×8 → 4×4)
        # ============================================================================
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)  # Very abstract representations
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # ============================================================================
        # BLOCK 6: Final feature extraction (4×4 → 1×1)
        # ============================================================================
        # Global feature aggregation - reduces spatial dimensions to 1×1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)  # Global receptive field
        self.B61 = nn.BatchNorm2d(128*hidden)

        # Average pooling (alternative to max pooling for smoother gradients)
        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # ============================================================================
        # FULLY CONNECTED LAYERS: Parameter prediction
        # ============================================================================
        # Final layers for parameter regression
        self.FC1  = nn.Linear(128*hidden, 64*hidden)  # Feature compression
        self.FC2  = nn.Linear(64*hidden,  12)         # Final prediction: 6 params + 6 uncertainties

        # ============================================================================
        # ACTIVATION FUNCTIONS AND REGULARIZATION
        # ============================================================================
        self.dropout   = nn.Dropout(p=dr)        # Prevent overfitting
        self.ReLU      = nn.ReLU()               # Standard activation
        self.LeakyReLU = nn.LeakyReLU(0.2)       # Allows small negative gradients
        self.tanh      = nn.Tanh()               # Bounded activation (-1, 1)

        # ============================================================================
        # WEIGHT INITIALIZATION
        # ============================================================================
        # Proper weight initialization for stable training
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)  # Scale parameter
                nn.init.constant_(m.bias, 0)    # Shift parameter
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        """
        Forward pass through the network.
        
        Args:
            image (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
                                 Expected input size: (N, channels, 256, 256)
        
        Returns:
            torch.Tensor: Predicted parameters and uncertainties of shape (batch_size, 12)
                         - indices 0-5: Parameter values (Ωₘ, σ₈, A_SN1, A_AGN1, A_SN2, A_AGN2)
                         - indices 6-11: Parameter uncertainties (σ_Ωₘ, σ_σ₈, σ_A_SN1, σ_A_AGN1, σ_A_SN2, σ_A_AGN2)
        
        Architecture Flow:
            Input (N, C, 256, 256) → Block0 → (N, 2H, 128, 128) → Block1 → (N, 4H, 64, 64) →
            Block2 → (N, 8H, 32, 32) → Block3 → (N, 16H, 16, 16) → Block4 → (N, 32H, 8, 8) →
            Block5 → (N, 64H, 4, 4) → Block6 → (N, 128H, 1, 1) → FC → (N, 12)
        """
        
        # ============================================================================
        # CONVOLUTIONAL FEATURE EXTRACTION
        # ============================================================================
        
        # Block 0: Initial feature extraction (256×256 → 128×128)
        x = self.LeakyReLU(self.C01(image))           # Conv + activation
        x = self.LeakyReLU(self.B02(self.C02(x)))     # Conv + batch norm + activation
        x = self.LeakyReLU(self.B03(self.C03(x)))     # Conv + downsample + batch norm + activation

        # Block 1: Feature development (128×128 → 64×64)
        x = self.LeakyReLU(self.B11(self.C11(x)))     # Expand channels to 4×hidden
        x = self.LeakyReLU(self.B12(self.C12(x)))     # Refine features
        x = self.LeakyReLU(self.B13(self.C13(x)))     # Spatial downsampling

        # Block 2: Mid-level abstraction (64×64 → 32×32)
        x = self.LeakyReLU(self.B21(self.C21(x)))     # Expand to 8×hidden channels
        x = self.LeakyReLU(self.B22(self.C22(x)))     # Feature processing
        x = self.LeakyReLU(self.B23(self.C23(x)))     # Spatial reduction

        # Block 3: High-level features (32×32 → 16×16)
        x = self.LeakyReLU(self.B31(self.C31(x)))     # Expand to 16×hidden channels
        x = self.LeakyReLU(self.B32(self.C32(x)))     # Abstract feature learning
        x = self.LeakyReLU(self.B33(self.C33(x)))     # Continue downsampling

        # Block 4: Abstract representations (16×16 → 8×8)
        x = self.LeakyReLU(self.B41(self.C41(x)))     # Expand to 32×hidden channels
        x = self.LeakyReLU(self.B42(self.C42(x)))     # Deep feature extraction
        x = self.LeakyReLU(self.B43(self.C43(x)))     # Spatial compression

        # Block 5: Very abstract features (8×8 → 4×4)
        x = self.LeakyReLU(self.B51(self.C51(x)))     # Expand to 64×hidden channels
        x = self.LeakyReLU(self.B52(self.C52(x)))     # Complex pattern recognition
        x = self.LeakyReLU(self.B53(self.C53(x)))     # Final spatial reduction

        # Block 6: Global feature aggregation (4×4 → 1×1)
        x = self.LeakyReLU(self.B61(self.C61(x)))     # Global receptive field, 128×hidden channels

        # ============================================================================
        # FULLY CONNECTED PARAMETER PREDICTION
        # ============================================================================
        
        # Flatten feature maps for fully connected layers
        x = x.view(image.shape[0], -1)  # Shape: (batch_size, 128*hidden)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # First fully connected layer with dropout
        x = self.dropout(self.LeakyReLU(self.FC1(x)))  # Shape: (batch_size, 64*hidden)
        
        # Final prediction layer
        x = self.FC2(x)  # Shape: (batch_size, 12)

        # ============================================================================
        # OUTPUT PROCESSING
        # ============================================================================
        
        # Ensure uncertainty estimates are positive by squaring
        # Clone tensor to avoid in-place operations that break gradients
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])  # Square uncertainty predictions (indices 6-11)
        
        # Final output:
        # y[:, 0:6]  - parameter predictions
        # y[:, 6:12] - uncertainty estimates (guaranteed positive)
        
        return y