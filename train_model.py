import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.metrics import confusion_matrix, accuracy_score
from test_model import do_test
# 3) TRAINING LOOP
def train_cgm_classifier(loader, val_loader, model, epochs=20,
                         lr=1e-3,mode=0):
    # prepare dataset + loader
# 1) Split into train / test
  

    model = model.to(device="cuda")
    criterion= nn.CrossEntropyLoss()
   
    optimizer= optim.Adam(model.parameters(), lr=lr)
    # total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params:      {total_params:,}")
    print(f"Trainable params:  {trainable_params:,}")
    model.train()
    min_val_loss = 100000
    for epoch in range(1, epochs+1):
        predictions = []
        labels = []
        probabilities = []
        running_loss = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(device="cuda")
            yb = yb.to(device="cuda")
            optimizer.zero_grad()
            probs = model(Xb)
            preds = list((torch.argmax(probs, axis=1)).float().cpu())
            predictions.extend(preds)
            labels.extend(yb.cpu())
            probabilities.extend(probs.detach().cpu())
            loss = criterion(probs, yb.squeeze().long())            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        #print(len(labels), len(predictions))
        #print(labels[0].size())
        #print(predictions[0].size())
        acc = accuracy_score(labels, predictions)
        avg_loss = running_loss / len(loader.dataset)
        #print(confusion_matrix(labels, predictions)) 
        
        #train_auc = roc_auc_score(labels, probabilities)

        
        probabilities, labels, val_loss = do_test(model, val_loader, criterion) 
        #auc = roc_auc_score(labels, probabilities)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} — Accuracy: {acc:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss<min_val_loss:
            print("Best Val Loss", val_loss)
            torch.save(model.state_dict(), "best_model_a1c"+str(mode)+".pth")
            min_val_loss = val_loss
    # return model and normalization stats (for inference)
    return probabilities
