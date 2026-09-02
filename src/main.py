import os
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as PyTorchDataLoader, TensorDataset

# Updated Import to match your project files
from dataloader import DataLoader as CustomDataLoader
from Shrink_Autoencoder import Shrink_Autoencoder
from client_trainer import ClientTrainer
from global_aggregator import GlobalAggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Shrink Autoencoder Training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs per client")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Shrink Autoencoder Specific Parameters
    parser.add_argument("--shrink_dim", type=int, default=16, help="Bottleneck/latent dimension")
    parser.add_argument("--shrink_threshold", type=float, default=0.2, help="Shrinkage operator threshold")

    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="FL algorithm")
    parser.add_argument("--fedprox_mu", type=float, default=0.01, help="Mu parameter for FedProx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load and process dataset
    logging.info(f"Loading data from {args.data_path}")
    data_processor = DataProcessor(args.data_path)
    client_data, dev_raw_data = data_processor.load_and_split(num_clients=args.num_clients)

    # 2. Correctly transform raw dev dataset to align scaling with clients
    logging.info("Preprocessing and transforming evaluation/dev data...")
    dev_transformed_data = data_processor.transform(dev_raw_data)
    dev_dataset = TensorDataset(torch.tensor(dev_transformed_data, dtype=torch.float32))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model Initialization using Shrink_Autoencoder
    input_dim = client_data[0].shape[1]
    global_model = Shrink_Autoencoder(
        input_dim=input_dim, 
        shrink_dim=args.shrink_dim, 
        threshold=args.shrink_threshold
    ).to(args.device)

    # 4. Initialize Global Aggregator and Local Clients
    global_aggregator = GlobalAggregator(global_model=global_model, dev_loader=dev_loader, device=args.device)

    clients = []
    for i in range(args.num_clients):
        transformed_client_data = data_processor.transform(client_data[i])
        client_dataset = TensorDataset(torch.tensor(transformed_client_data, dtype=torch.float32))
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)

        client_save_dir = os.path.join(args.save_dir, f"client_{i}")
        client_model = Shrink_Autoencoder(
            input_dim=input_dim, 
            shrink_dim=args.shrink_dim, 
            threshold=args.shrink_threshold
        )
        
        client = ClientTrainer(
            client_id=i,
            model=client_model,
            train_loader=client_loader,
            epochs=args.local_epochs,
            lr=args.lr,
            algorithm=args.algorithm,
            fedprox_mu=args.fedprox_mu,
            device=args.device,
            save_dir=client_save_dir
        )
        clients.append(client)

    # 5. Federated Training Loop
    logging.info("Starting Federated Training with Shrink Autoencoder...")
    for round_num in range(1, args.num_rounds + 1):
        logging.info(f"--- Round {round_num}/{args.num_rounds} ---")
        client_weights = []
        client_data_sizes = []

        # Send global parameters to clients
        global_params = global_aggregator.get_global_parameters()

        for client in clients:
            client.set_parameters(global_params)
            loss = client.train()
            client_weights.append(client.get_parameters())
            client_data_sizes.append(len(client.train_loader.dataset))

        # Aggregate parameters on server
        global_aggregator.aggregate(client_weights, client_data_sizes)

        # Evaluate performance on evaluation dev set
        dev_loss = global_aggregator.evaluate()
        logging.info(f"Round {round_num} Global Dev Loss: {dev_loss:.6f}")

    # Save final aggregated global model
    global_save_path = os.path.join(args.save_dir, "global_shrink_autoencoder.pt")
    torch.save(global_model.state_dict(), global_save_path)
    logging.info(f"Training Complete. Saved global model to {global_save_path}")


if __name__ == "__main__":
    main()
