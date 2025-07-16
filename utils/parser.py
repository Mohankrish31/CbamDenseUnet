import argparse
import json
import os
def get_config():
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Configuration")
    # Required
    parser.add_argument('--config', type=str, required=True, help='Path to the config JSON file')
    # Optional overrides
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'val', 'test'],
                        help='Operation mode to override config')
    parser.add_argument('--device', type=str, default=None, help='Override device (e.g. cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    # Load the config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    # Optional overrides
    if args.device:
        for key in ['train', 'val', 'test']:
            if key in config:
                config[key]['device'] = args.device
    if args.batch_size:
        for key in ['train', 'val', 'test']:
            if key in config and 'dataloader' in config[key]:
                config[key]['dataloader']['args']['batch_size'] = args.batch_size
    if args.mode:
        config['run_mode'] = args.mode  # Add mode to config for downstream usage
    return config
