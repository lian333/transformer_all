import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.timefeatures import time_features

from sklearn.preprocessing import StandardScaler
import os

def load_data(args):
    datapath=os.path.join(args.root_path,args.data_path)
    df = pd.read_csv(datapath)
    df=df
    df.drop_duplicates(subset=[df.columns[0]], inplace=True)
    df.rename(columns={'dt_iso':'date'},inplace=True)

    df.drop([df.columns[1]], axis=1, inplace=True)
    # weather_main
    listType = df['weather_main'].unique()
    df.ffill(inplace=True)
    dic = dict.fromkeys(listType)
    for i in range(len(listType)):
        dic[listType[i]] = i
    df['weather_main'] = df['weather_main'].map(dic)
    # weather_description
    listType = df['weather_description'].unique()
    dic = dict.fromkeys(listType)
    for i in range(len(listType)):
        dic[listType[i]] = i
    df['weather_description'] = df['weather_description'].map(dic)
    # weather_icon
    listType = df['weather_icon'].unique()
    dic = dict.fromkeys(listType)
    for i in range(len(listType)):
        dic[listType[i]] = i
    df['weather_icon'] = df['weather_icon'].map(dic)
    # print(df)
    return df

def get_data(args):
    print('data processing...')
    data = load_data(args)
    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    scaler = StandardScaler()

    def process(dataset, flag, step_size, shuffle):
        # 对时间列进行编码
        df_stamp = dataset['date']
        df_stamp.date = pd.to_datetime(df_stamp)
        data_stamp = time_features(df_stamp, timeenc=1, freq=args.freq)
        data_stamp = torch.FloatTensor(data_stamp)
        # 接着归一化
        # 首先去掉时间列
        dataset = dataset.copy()

        dataset.drop(['date'], axis=1, inplace=True)
        if flag == 'train':
            dataset = scaler.fit_transform(dataset.values)
        else:
            dataset = scaler.transform(dataset.values)

        dataset = torch.FloatTensor(dataset)
        # 构造样本
        samples = []
        # print('dataset shape',len(dataset))
        # print('args.seq_len',args.seq_len)
        # print('args.pred_len',args.pred_len)
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

        samples = DataLoader(dataset=samples, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=True)

        
        return samples

    Dtr = process(train, flag='train', step_size=1, shuffle=True)
    Val = process(val, flag='val', step_size=1, shuffle=True)
    Dte = process(test, flag='test', step_size=args.pred_len, shuffle=False)

    
    print('the number of the batch in training data:',int(len(Dtr.dataset)/args.batch_size))
    print('Training data shape',len(Dtr.dataset))
    print('Validation data shape',len(Val.dataset))
    print('Testing data shape',len(Val.dataset))
    for x, y, z, f in Dtr:
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(f.shape)
        break
        
    return Dtr, Val, Dte, scaler

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
        enc_in=7, dec_in=7, c_out=7, freq='h', d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=0.5, dropout=0.05, attn='prob', embed='timeF', activation='gelu',
        output_attention=False, distil=True, mix=True, padding=0, seq_len=96, label_len=48,
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

    print(x[0][71])
    print(y[0][71])