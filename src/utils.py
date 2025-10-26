import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_repo_home():
    """
    Get the repository home directory path.
    
    Returns:
        str: Absolute path to the repository root directory
    """
    # Get the absolute path of the current file
    current_file = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()
    
    # Navigate up to find the repo root (CNN2D-Cosmology-Interpretability)
    current_dir = os.path.dirname(current_file) if os.path.isfile(current_file) else current_file
    
    while current_dir != '/' and not current_dir.endswith('CNN2D-Cosmology-Interpretability'):
        current_dir = os.path.dirname(current_dir)
    
    # If we can't find the repo root, use the current working directory
    if not current_dir.endswith('CNN2D-Cosmology-Interpretability'):
        # Try to find it in the current working directory path
        cwd = os.getcwd()
        if 'CNN2D-Cosmology-Interpretability' in cwd:
            parts = cwd.split('CNN2D-Cosmology-Interpretability')
            current_dir = parts[0] + 'CNN2D-Cosmology-Interpretability'
        else:
            # Fallback: assume we're in the repo somewhere
            current_dir = '/Users/arnablahiry/repos/CNN2D-Cosmology-Interpretability'
    
    return current_dir

def expand_path(path):
    """
    Expand a path that starts with ~ to use the repo home directory.
    
    Args:
        path (str): Path starting with ~ (repo home) or absolute path
        
    Returns:
        str: Expanded absolute path
    """
    if path.startswith('~'):
        repo_home = get_repo_home()
        return path.replace('~', repo_home, 1)
    return path