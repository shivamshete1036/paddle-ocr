import os
import sys

# Ensure the system knows where our custom utils are
sys.path.append('./')

# Disable GPU and OneDNN to prevent the crash we faced earlier
os.environ['FLAGS_use_onednn'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from paddleocr.tools.train import main
from paddleocr.utils.utility import parse_args

if __name__ == '__main__':
    # This script acts as a wrapper for the official PaddleOCR training engine
    # It takes the config file (-c) and runs the fine-tuning logic
    
    # Check if config exists
    config_path = 'custom_finetune.yml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found! Create it first.")
        sys.exit(1)

    print("--- Initializing PaddleOCR Fine-Tuning ---")
    print(f"Using Config: {config_path}")
    
    # We simulate the command line arguments
    sys.argv = ['train.py', '-c', config_path]
    
    try:
        main()
    except Exception as e:
        print(f"\nTraining interrupted or failed: {e}")