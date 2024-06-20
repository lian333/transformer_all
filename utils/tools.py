import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,IterableDataset,ConcatDataset,TensorDataset
import pickle
from pyod.models.ecod import ECOD
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
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



def plotbatch(df):

    fig, axs = plt.subplots(len(df.columns), 1, figsize=(10, 2*len(df.columns)))
    timetitle=df.date.to_list()
    timetitle=timetitle[0]
    timetitle_str = str(timetitle).replace(':', '-')

    for i, count in enumerate(df.columns):
        axs[i].plot(df[count], label=count)
        axs[i].set_title(count)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，给 suptitle 留出空间

    fig.suptitle(timetitle, fontsize=12)
    output_path = f'D:/studydata/Masterarbeit/fullpicture_more10000_axis1/{timetitle_str}.png'

    plt.savefig(output_path)
    plt.close(fig)  # close the image 
    return output_path


def detect_outliers(data):
       # Convert data to float for compatibility
    data = data.astype(float)
    # Isolation Forest
    X = data.values.reshape(-1, 1)
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    clf.fit(X)
    outliers_if = clf.predict(X)
    outliers_if = np.where(outliers_if == -1)[0]

    # DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=3)
    outliers_dbscan = dbscan.fit(X).labels_
    outliers_dbscan = np.where(outliers_dbscan == -1)[0]

    # ECOD
    ecod = ECOD(contamination=0.15)
    outliers_ecod = ecod.fit(X).labels_
    outliers_ecod = np.where(outliers_ecod == 1)[0]
    # if outliers are detected two or more times, keep them
    outliers_if = outliers_if
    outliers_dbscan = outliers_dbscan
    outliers_ecod = outliers_ecod
    outliers = np.concatenate((outliers_if, outliers_dbscan, outliers_ecod))
    outliers = list(outliers)
    index=[]
    for x in outliers:
        if outliers.count(x) >= 3:
           index.append(x)
    # 移动平滑替换异常值，使用窗口大小为20
    plotdata_smooth = data.copy()
    plotdata_smooth = plotdata_smooth.astype(float)

    for i in index:
        if i<5:
            plotdata_smooth[i] = np.mean(data[i:i+5]).astype(float)

        elif i>len(plotdata_smooth)-5:
            plotdata_smooth[i] = np.mean(data[i-5:i]).astype(float)
        else:
            plotdata_smooth[i] = np.mean(data[i-5:i+5]).astype(float)
    
    return  plotdata_smooth,index

# plot the outliers
def switch(testdf,plot):
    # Ensure all data columns (except 'date') are float type for compatibility
    testdf = testdf.apply(lambda x: x.astype(float) if x.name != 'date' else x)
    fig, axs = plt.subplots(len(testdf.columns[1:]), 1, figsize=(10, 2*len(testdf.columns[1:])))

    timetitle=list(testdf['date'])[0]
    timetitle_str = str(timetitle).replace(':', '-')
    smoothed_data  = testdf.copy()

    for i, column in enumerate(testdf.columns[2:]):
        plotdata=testdf[column].reset_index(drop=True) 
        plotdata_smooth, outliers = detect_outliers(plotdata)
        smoothed_data[column] = plotdata_smooth.values
        if plot:

            axs[i].plot(plotdata, label='Original',zorder=2)
            axs[i].scatter(outliers, plotdata[outliers], color='red', label='Outliers',zorder=1)
            axs[i].plot(plotdata_smooth, label='Smoothed',zorder=3)
            axs[i].legend()
    if plot:

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，给 suptitle 留出空间
        fig.suptitle(timetitle, fontsize=16)
        output_path = f'D:/studydata/Masterarbeit/fullpicture_more10000_axis1/{timetitle_str}_outlier.png'
        plt.savefig(output_path)
        plt.close(fig)

    return testdf
