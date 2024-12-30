import argparse
from data_loader import load_data
from model import create_model
from train import train_model
from utils import setup_logging, setup_device

def main(args):
    # Setup logging and device
    logger = setup_logging(args.log_level)
    device = setup_device(args.device)

    logger.info(f"Using device: {device}")

    # Load and preprocess data
    train_loader, test_loader = load_data(args.framework, args.batch_size)

    # Create model
    model = create_model(args.framework).to(device)

    # Train model
    train_model(args.framework, model, train_loader, test_loader, device, args.epochs, args.lr)

    # Save the model
    if args.framework == 'tensorflow':
        model.save('models/tf_model')
    elif args.framework == 'pytorch':
        import torch
        torch.save(model.state_dict(), 'models/torch_model.pth')

    logger.info("Training completed and model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model using TensorFlow or PyTorch')
    parser.add_argument('--framework', type=str, choices=['tensorflow', 'pytorch'], default='tensorflow',
                        help='Choose the framework to use (tensorflow or pytorch)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()
    
    main(args)

