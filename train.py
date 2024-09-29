# PyTorch Lightning and callbacks
import pytorch_lightning as pl
# Miscellaneous
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
# Datasets and data handling
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from dataset import get_tokenized_dataset_form_json, split_dataset
from model import get_model
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(model, 
          checkpoint_callback, 
          train_dataloader, 
          validation_dataloader, 
          epochs = 10, 
          strategy = "auto",
          devices = 2,
          num_nodes = 2):
    # Tensorboard
    tensorboard_logger = TensorBoardLogger("logs")#May be useless
    # Create the trainer
    trainer = pl.Trainer(
        max_epochs=epochs,                       # Number of epochs
        callbacks=[checkpoint_callback],         # Model checkpoint callback
        logger=tensorboard_logger,               # TensorBoard logger
        strategy=strategy,                       # Distributed data parallel strategy (ddp)
        devices=devices,                               # Number of GPUs per node
        num_nodes=num_nodes,                             # Number of nodes
    )
    # Start training
    trainer.fit(model, train_dataloader, validation_dataloader)

def main(model_name = "bert-base-cased", 
         json_file = "dev-v2.0.json", 
         strategy = "auto", 
         epochs = 10, 
         devices = 2, 
         nodes = 2,
         use_tensor_cores = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    # Load the dataset
    tokenized_dataset = get_tokenized_dataset_form_json(model_name, json_file)
    tokenized_dataset_train, tokenized_dataset_validation = split_dataset(tokenized_dataset)
    tokenized_dataset_train.set_format(type='torch')# To be sure to have tensors
    tokenized_dataset_validation.set_format(type='torch')# TO be sure to have tensors

    # To ensure to always have the same results
    train_dataloader = DataLoader(dataset=tokenized_dataset_train, batch_size=16, num_workers=31)
    validation_dataloader = DataLoader(dataset=tokenized_dataset_validation, batch_size=16, num_workers=31)

    # Load the model
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # Create the optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    # Create the model
    model = get_model(model, optim)

    # Model checkpoint callback for saving checkpoints every 2 epochs
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=2,  # Save every 2 epochs
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=True  # Save a model or a checkpoint
    )
    if (use_tensor_cores):
        # Use the best of A100
        torch.set_float32_matmul_precision('high')  # High to get the best results (could test with medium) desactivate for now (too fast)


    # Train the model
    train(model, 
          checkpoint_callback, 
          train_dataloader, 
          validation_dataloader, 
          strategy=strategy, 
          epochs=epochs,
          devices = devices,
          num_nodes = nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of the model", default="bert-base-cased")
    parser.add_argument("--json", help="json file path", default="dev-v2.0.json")
    parser.add_argument("--strategy", help="json file path", default="auto")# auto, ddp, fsdp
    parser.add_argument("--epochs", help="epochs", default=10, type=int)
    parser.add_argument("--devices", help="number of devices", default=2, type=int)
    parser.add_argument("--nodes", help="number of nodes", default=2, type=int)
    #parser.add_argument("--use_tensor_cores", help="use tensor cores of A100", default=False, type=bool)
    parser.add_argument("--use_tensor_cores", help="use tensor cores of A100", action='store_true', default=False)
    args = parser.parse_args()
    main(args.model, 
         args.json, 
         args.strategy, 
         args.epochs, 
         args.devices, 
         args.nodes, 
         args.use_tensor_cores)


"""
The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.
torch.set_float32_matmul_precision('high')  # High to get the best results (could test with medium) desactivate for now (too fast)


srun --gres=gpu:4 --time=01:00:00 --mem=16G -c 48 --pty --exclusive bash
./train.sh
"""
