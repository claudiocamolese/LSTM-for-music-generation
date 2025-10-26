import torch
import os
from tqdm import tqdm


def train(model, train_loader, val_loader, config, experiment=None, device='cpu'):
    """
    Trains an LSTM-based model for sequence prediction (e.g., music generation).

    Args:
        model (nn.Module): 
            The PyTorch model to train. Must implement a `forward` method returning (outputs, hidden),
            and an `init_hidden(batch_size)` method for initializing hidden states.
        
        train_loader (DataLoader): 
            DataLoader providing the training data as batches of (inputs, targets).
        
        val_loader (DataLoader): 
            DataLoader providing the validation data as batches of (inputs, targets).
        
        config (dict): 
            Dictionary containing training hyperparameters. Expected keys:
                - 'train': {
                    'flag' : bool,              # Decide to do or skip the train
                    'lr': float,                # learning rate for the optimizer
                    'num_epochs': int,          # number of training epochs
                    'batch_size': int,          # batch size used in training
                    'clip': float               # max gradient norm for clipping
                    'test_size':                # in which percentage devide the dataset
                  }
        
        experiment (optional): 
            Experiment tracking object (e.g., Comet ML or similar) used to log metrics during training.
            Default is None.
        
        device (str, optional): 
            Device on which to train the model ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None:
            The function trains the model, evaluates it on the validation set at each epoch,
            logs relevant metrics (if an experiment tracker is provided),
            and saves the best-performing model (lowest validation loss) in the './models' directory.
    """
    
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss() # -log(softmax(x)[y])
    optimizer = torch.optim.Adam(model.parameters(), lr= float(config['train']['lr']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer= optimizer, 
                                                           T_max= config["train"]["num_epochs"], 
                                                           eta_min=1e-8)

    num_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    
    best_val_loss = float('inf')
    best_model_state = None
    step = 1

    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        total_loss = 0
        
        # for each epoch initialize the hidden state
        hidden = model.init_hidden(batch_size)

        for inputs, targets in tqdm(train_loader, desc="Training batch", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # for each batch, initialize the hidden state
            hidden = model.init_hidden(batch_size)
            
            model.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= config['train']['clip'])

            optimizer.step()
            total_loss += loss.item()
            
            if experiment:
                experiment.log_metric("Batch Loss", loss.item())
                experiment.log_metric("Learning Rate", scheduler.get_last_lr()[0])
                experiment.log_metric("Total Loss", total_loss)
            
            step += 1
                

        avg_train_loss = total_loss / len(train_loader)
        
        if experiment:
            experiment.log_metric("Average epoch loss", avg_train_loss)
        

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation batch", unit="batch"):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                
                # initialize hidden state 
                val_hidden = model.init_hidden(batch_size)   
                
                outputs, val_hidden = model(inputs, val_hidden)
                loss = criterion(outputs, targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if experiment:
            experiment.log_metric("Train Loss", avg_train_loss, step=step)
            experiment.log_metric("Validation Loss", avg_val_loss, step=step)
            experiment.log_metric("Learning Rate", scheduler.get_last_lr()[0], step=step)

        
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
                # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            
            save_dir = os.path.join('music_lstm', 'models')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            print("Saved new model")

        

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        