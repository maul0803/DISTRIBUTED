import pytorch_lightning as pl
from metrics import compute_metric
import torch

class MyModel(pl.LightningModule):
    def __init__(self, model, optim, compute_metric):
        super(MyModel, self).__init__()
        self.model = model
        self.optim = optim
        self.compute_metric = compute_metric
        
        self.train_losses = []
        self.train_f1s = []
        self.val_losses = []
        self.val_f1s = []

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        return self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                          end_positions=end_positions)

    def configure_optimizers(self):
        return self.optim

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)

        # Forward pass
        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs.loss
        # Compute metrics
        f1 = self.compute_metric(outputs, input_ids, start_positions, end_positions)
        f1 = torch.tensor(f1).to(self.device)
        # Log loss and metrics
        #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss)
        self.train_f1s.append(f1)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)

        # Forward pass
        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs.loss
        # Compute metrics
        f1 = self.compute_metric(outputs, input_ids, start_positions, end_positions)
        f1 = torch.tensor(f1).to(self.device)

        # Log validation loss
        #self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_losses.append(loss)
        self.val_f1s.append(f1)

        return {'val_loss': loss, 'val_f1': f1}

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        avg_f1 = torch.stack(self.train_f1s).mean()

        # Log average loss and metrics
        self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1_epoch', avg_f1, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.clear()
        self.train_f1s.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        avg_val_f1 = torch.stack(self.val_f1s).mean()

        # Log average validation loss and metrics
        self.log('val_loss_epoch', avg_val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1_epoch', avg_val_f1, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_losses.clear()
        self.val_f1s.clear()

def get_model(model, optim):
    return  MyModel(model, optim, compute_metric)