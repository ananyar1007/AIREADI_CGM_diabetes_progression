import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from test_model import do_test
import numpy as np
# 3) TRAINING LOOP
def train_cgm_classifier(loader, val_loader, model, epochs=20,
                         lr=1e-3,mode=0):
    # prepare dataset + loader
# 1) Split into train / test
  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ----- Compute class weights from dataset -----
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.cpu().numpy().astype(int).flatten())

    # Count samples per class
    #print(all_labels)
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / (class_counts + 1e-6)  # avoid divide by zero
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize

    # Convert to tensor and move to GPU
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ----- Define weighted loss -----
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ----- Optimizer -----
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    print(f"Class Weights: {class_weights.cpu().numpy()}")
    model.train()
    min_val_loss = 100000
    max_val_ba = 0
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

        
        acc = accuracy_score(labels, predictions)
        avg_loss = running_loss / len(loader.dataset)
  
        
        probabilities, labels, val_loss = do_test(model, val_loader, criterion) 
        # Convert to NumPy arrays
        probs = np.vstack(probabilities)  # shape (N, 2)
        labels = np.array([int(l.item()) for l in labels])  # shape (N,)

        # Extract probability for the positive class (class 1)
        y_prob = probs[:, 1]
        y_true = labels
        y_pred = (y_prob > 0.5).astype(int)

        val_ba = balanced_accuracy_score(y_true, y_pred)        #auc = roc_auc_score(labels, probabilities)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} — Accuracy: {acc:.4f} - Val Loss: {val_loss:.4f}")
        #if val_ba > max_val_ba:
        if val_loss<min_val_loss:
            print("Best Val Loss", val_loss)
#            print("Best Val ba", val_ba)
            best_model = model
            min_val_loss = val_loss
            #max_val_ba = val_ba
    # return model and normalization stats (for inference)
    return probabilities, best_model
