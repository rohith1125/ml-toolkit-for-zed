import logging
import torch

def setup_logging(log_level):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def setup_device(device):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

