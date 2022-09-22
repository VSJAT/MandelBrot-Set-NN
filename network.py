#Pytorch Lightning Model

from torch import optim
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


class DeepMLP(LightningModule):
    def __init__(self, in_size, hidden_size, hidden_layers, out_size, learning_rate, optimizer, train_loader, val_loader, test_loader):
        super().__init__()

        self.learning_rate = learning_rate # Learning rate
        self.loss_fn = nn.MSELoss() 
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


        self.modules = [nn.Linear(in_size, hidden_size), nn.ReLU()]

        for _ in range(hidden_layers):
            
            self.modules.append(nn.Linear(hidden_size, hidden_size))
            self.modules.append(nn.ReLU())
        
        self.modules.append(nn.Linear(hidden_size, out_size))
        #Since we are using MSELoss we don't need to use Sigmoid as MSELoss does it for us

        self.model = nn.Sequential(*self.modules)

        
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        # Return Adam or SGD optimizer
        if self.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
            print("SGD")
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            print("ADAM")
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
    
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
