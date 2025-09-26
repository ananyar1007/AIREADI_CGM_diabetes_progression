import torch

def do_test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total   = 0
    predictions = []
    probabilities = []
    labels = [] 
    running_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device="cuda")
            yb = yb.to(device="cuda")
            probs = model(Xb)
            preds = (torch.argmax(probs, axis=1)).float().cpu()
            probabilities.extend(probs.cpu().numpy())
            labels.extend(yb.cpu())
            predictions.extend(list(preds))
            correct += (preds == yb.cpu()).sum().item()
            total   += yb.numel()
            loss = criterion(probs, yb.squeeze().long()) 
            running_loss += loss.item() * Xb.size(0) 

    avg_loss = running_loss / len(test_loader.dataset)
    return probabilities, labels, avg_loss