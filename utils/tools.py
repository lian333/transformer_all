import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,IterableDataset,ConcatDataset,TensorDataset
import pickle

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def gettime():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())




class CustomDataset(Dataset):
    def __init__(self, filename, scaler):
        self.filename = filename
        self.scaler = scaler
        self.batch_indices = self._get_batch_indices()
    def _get_batch_indices(self):
        batch_indices = []
        with open(self.filename, 'rb') as f:
            pickle.load(f)  # 读取 scaler 数据并跳过
            while True:
                try:
                    batch_start = f.tell()
                    pickle.load(f)
                    batch_indices.append(batch_start)
                except EOFError:
                    break
        return batch_indices

    def __len__(self):
        return len(self.batch_indices)
    
    def __getitem__(self, idx):
        with open(self.filename, 'rb') as f:
            f.seek(self.batch_indices[idx])
            batch = pickle.load(f)
        return batch
    
    def inverse_transform(self, data):
        min_val = torch.from_numpy(self.scaler.data_min_).type_as(data).to(data.device)
        max_val = torch.from_numpy(self.scaler.data_max_).type_as(data).to(data.device)
        
        if data.shape[-1] != min_val.shape[-1]:
            min_val = min_val[-1:]
            max_val = max_val[-1:]

        return data * (max_val - min_val) + min_val



def load_dataloader_and_scaler(filename):
    with open(filename, 'rb') as f:
        saved_dict = pickle.load(f)
    # batches = saved_dict['batches']
    scaler = saved_dict['scaler']
    
    custom_dataset = CustomDataset(filename,scaler)
    dataloader = DataLoader(dataset=custom_dataset, batch_size=None, shuffle=False)
    print('Dataloader and Scaler loaded successfully!')

    return dataloader, scaler
