import torch
from torch.utils.data import Dataset, DataLoader,IterableDataset,ConcatDataset,TensorDataset
import pandas as pd
from utils.timefeatures import time_features
from  utils.tools import plotbatch,detect_outliers,switch,load_dataloader_and_scaler
import math
import json
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from datetime import timedelta
import logging
import random



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# Custom Dataset class to match the expected output structure
class CustomDataset(Dataset):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
def load_data(args):
    datapath=os.path.join(args.root_path,args.data_path)
    synthpath=os.path.join(args.root_path,args.synthetic_data)
    #datapath=r'D:\studydata\Masterarbeit\Data\all_data_axis1.csv'
    #datapath=r'D:\studydata\Masterarbeit\lian333\informer\data\ETT\axis2_demo_tablepart1_0.csv'
    df = pd.read_csv(datapath)
    df=df
    df.drop_duplicates(subset=[df.columns[0]], inplace=True)
    jsonfile=os.path.join(args.root_path,args.feature_path)
    with open(jsonfile, 'r') as file:
        features = json.load(file)[:10]
    df["date"] = pd.to_datetime(df.Timestamp, unit='s')
    features=["date",'Schadensklasse']+features
    data_selected = df[features]
    data_selected

    if args.synthetic:
        synth_df = pd.read_csv(synthpath)
        synth_df["date"] = pd.to_datetime(synth_df.Timestamp, unit='s')
        synth_df['Schadensklasse'] = 1
        synth_df = synth_df[features]
        data_selected = pd.concat([data_selected, synth_df], axis=0, ignore_index=True)

    return data_selected
    # datapath=os.path.join(args.root_path,args.data_path)

    # axis1df = pd.read_csv(datapath)
    # axis1df["date"] = pd.to_datetime(axis1df.Timestamp, unit='s')
    # if True:

    #     axis1df['Schadensklasse'] = 1

    # axis1df = axis1df[features]
    # axis1df
    # combined_df = pd.concat([data_selected, axis1df], axis=0, ignore_index=True)

    # df.rename(columns={'datetime':'date'},inplace=True)
    # data_selected.to_csv('true_data.csv', index=False)

def get_data(args):
    print('data processing...')
    data = load_data(args)
    # split
    alltime=np.unique(data.date)
    after=[]
    for x in alltime:
        length=len(data[data["date"]==x])
        damage=np.unique(data[data["date"]==x].Schadensklasse)
        # if length >10000 and len(damage)>=2:

        if length >=args.length:
            after.append(x)
            print('length of batches: %d',length)
    print('='*30)



    scaler = MinMaxScaler()
    data = data.copy()
    data_to_scale =data.drop(['date',"Schadensklasse"], axis=1, inplace=False)

    scaler.fit(data_to_scale)


    def process(dataset, flag, step_size, shuffle):
            # 对时间列进行编码
            df_stamp = dataset['date']
            df_stamp.date = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, timeenc=1, freq=args.freq)
            data_stamp = torch.FloatTensor(data_stamp)
            # 接着归一化
            # 首先去掉时间列
            dataset = dataset.copy()

            dataset.drop(['date',"Schadensklasse"], axis=1, inplace=True)
            if flag == 'train':
                dataset = scaler.transform(dataset.values)
            else:
                dataset = scaler.transform(dataset.values)

            dataset = torch.FloatTensor(dataset)
            # 构造样本
            samples = []
            #  print('dataset shape',len(dataset))
            #  print('args.seq_len',args.seq_len)
            count = 0

            for index in range(0, len(dataset) - args.seq_len - args.pred_len + 1, step_size):
                # train_x, x_mark, train_y, y_mark
                s_begin = index
                s_end = s_begin + args.seq_len
                r_begin = s_end - args.label_len
                r_end = r_begin + args.label_len + args.pred_len
                seq_x = dataset[s_begin:s_end]
                seq_y = dataset[r_begin:r_end]
                seq_x_mark = data_stamp[s_begin:s_end]
                seq_y_mark = data_stamp[r_begin:r_end]
                samples.append((seq_x, seq_y, seq_x_mark, seq_y_mark))
                count += 1

            print('number of dataset in one batch: %d', count)
            samples = DataLoader(dataset=samples, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=True)

            return samples
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(42)

    Dtr_samples = []
    Val_samples = []
    Dte_samples = []


    for time in after:
        finaldata=data[data["date"]==time]
        start_date = finaldata['date'].iloc[0]
        # 生成一个时间序列，从起始日期开始，每秒递增，长度与data相同
        time_series = [start_date + timedelta(seconds=i) for i in range(len(finaldata))]

        # 将生成的时间序列赋值回data的'date'列
        finaldata.loc[:, 'date'] = time_series
        # path=plotbatch(finaldata)
        # finaldata=switch(finaldata, False)
        # print(data)
        train = finaldata[:int(len(finaldata) * 0.6)]
        val = finaldata[int(len(finaldata) * 0.6):int(len(finaldata) * 0.8)]
        test = finaldata[int(len(finaldata) * 0.8):len(finaldata)]
        print('length of the training  dataset after times 0.6: %d', len(train))
        print('length of the validation dataset after times 0.2: %d', len(val))
        print('length of the testing dataset after times 0.2: %d', len(test))
        Dtr = process(train, flag='train', step_size=1, shuffle=True)
        Val = process(val, flag='val', step_size=1, shuffle=True)
        Dte = process(test, flag='test', step_size=args.pred_len, shuffle=False)
        print('='*30)

        Dtr_samples.append(Dtr)
        print(len(Dtr_samples))
        Val_samples.append(Val)
        Dte_samples.append(Dte)


    Dtr_datasets = [CustomDataset(dataloader) for dataloader in Dtr_samples]
    Dtr_combined_dataset = ConcatDataset(Dtr_datasets)
    del Dtr_datasets
    Dtr_combined_dataloader = DataLoader(Dtr_combined_dataset, batch_size=32, shuffle=True)
    print("finish  training data after dataloader with batch_size",len(Dtr_combined_dataloader))

    Dte_datasets = [CustomDataset(dataloader) for dataloader in Dte_samples]
    Dte_combined_dataset = ConcatDataset(Dte_datasets)
    del Dte_datasets
    Dte_combined_dataloader = DataLoader(Dte_combined_dataset, batch_size=32, shuffle=False)

    Val_datasets = [CustomDataset(dataloader) for dataloader in Val_samples]
    Val_combined_dataset = ConcatDataset(Val_datasets)
    del Val_datasets
    Val_combined_dataloader = DataLoader(Val_combined_dataset, batch_size=32, shuffle=True)

    print("x"*30)



    # load_combined_dataloader, scaler = load_dataloader_and_scaler(valid_pickel_file)
    
    # load_combined_dataloader, scaler = load_dataloader_and_scaler(valid_pickel_file)

    print('The length of training dataset after Concatenation  %d',len(Dtr_combined_dataloader.dataset))
    for count,(x ,y,z,f) in enumerate(Dtr_combined_dataloader):
        if count==0:
             print('one complete training batch shape: %s',x.shape)
             print(x[0][0])

             print('one complete training batch shape: %s',y.shape)
             print('one complete training batch shape: %s',z.shape)
             print('one complete training batch shape: %s',f.shape)
    print('final training batch shape:%s',x.shape)
    print('final training batch shape:%s',y.shape)
    print('final training batch shape:%s',z.shape)
    print('final training batch shape:%s',f.shape)
    print('the count of batches in training dataset:%d',count+1)
    print('calcation process: (count-1)*batch size + final batch shape = %d',len(Dtr_combined_dataloader.dataset))
    print('='*30)

    print('The length of validation dataset after Concatenation: %d',len(Val_combined_dataloader.dataset))
    for count,(x ,y,z,f) in enumerate(Val_combined_dataloader):
        if count==0:
             print('one complete validation batch shape:%s',x.shape)
             print('one complete validation batch shape:%s',y.shape)
             print('one complete validation batch shape:%s',z.shape)
             print('one complete validation batch shape:%s',f.shape)
    print('final validation batch shape:%s',x.shape)
    print('final validation batch shape:%s',y.shape)
    print('final validation batch shape:%s',z.shape)
    print('final validation batch shape:%s',f.shape)
    print('the count of batches in validation dataset:%d',count+1)
    print('calcation process: (count-1)*batch size + final batch shape = %d',len(Val_combined_dataloader.dataset))
    print('='*30)


    print('Dtr data shape',len(Dtr_combined_dataloader.dataset))
    print('Val data shape',len(Val_combined_dataloader.dataset))
    print('Dte data shape',len(Dte_combined_dataloader.dataset))
  
    return Dtr_combined_dataloader, Val_combined_dataloader, Dte_combined_dataloader, scaler

if __name__ == '__main__':



    class Args:
        def __init__(self, enc_in, dec_in, c_out, freq, d_model, n_heads, e_layers, d_layers, d_ff, factor, dropout,
                    attn, embed, activation, output_attention, distil, mix, padding, seq_len, label_len, pred_len, 
                    device, batch_size, epochs, lr, lambd, patience, des):
            self.enc_in = enc_in
            self.dec_in = dec_in
            self.c_out = c_out
            self.freq = freq
            self.d_model = d_model
            self.n_heads = n_heads
            self.e_layers = e_layers
            self.d_layers = d_layers
            self.d_ff = d_ff
            self.factor = factor
            self.dropout = dropout
            self.attn = attn
            self.embed = embed
            self.activation = activation
            self.output_attention = output_attention
            self.distil = distil
            self.mix = mix
            self.padding = padding
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            self.device = device
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.lambd = lambd
            self.patience = patience
            self.des = des

    # Example usage:
    args = Args(
        enc_in=7, dec_in=7, c_out=7, freq='s', d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=0.5, dropout=0.05, attn='prob', embed='timeF', activation='gelu',
        output_attention=False, distil=True, mix=True, padding=0, seq_len=96, label_len=96,
        pred_len=24, device='cuda', batch_size=32, epochs=10, lr=1e-3, lambd=1, patience=5, des='test'
    )

    # change the args format,let  args.pred_len work
    Dtr, Val, Dte, scaler=get_data(args=args)

    for x, y, z, f in Dtr:
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(f.shape)
        break

    print(z[12][95])
    print(f[12][95])