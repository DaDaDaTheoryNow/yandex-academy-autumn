import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, precision, recall


def run_epoch(device, model, opt, loss_fn, dataloader, is_train=True, return_predictions=False):
    model.train(is_train)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.set_grad_enabled(is_train):
        for x, label in tqdm(dataloader, desc="Train" if is_train else "Val"):
            x = x.to(device)
            label = label.to(device).long()
            
            pred = model(x)
            l = loss_fn(pred, label)
            
            if is_train:
                opt.zero_grad()
                l.backward()
                opt.step()
            
            total_loss += l.item()
            
            if return_predictions:
                all_preds.extend(pred.argmax(dim=1).cpu().numpy())
                all_labels.extend(label.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    if return_predictions:
        f1, precision, recall = calculate_metrics(all_labels, all_preds)
        return avg_loss, f1, precision, recall
    return avg_loss


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
