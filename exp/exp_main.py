import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer,LSTM,GRU
from utils.tools import EarlyStopping, adjust_learning_rate, gettime,load_dataloader_and_scaler
from utils.metrics import metric
from get_data import get_data

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
import os
import matplotlib.pyplot as plt
import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.train_loader, self.vali_loader, self.test_loader, self.scaler = get_data(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'LSTM':LSTM,
            'GRU':GRU
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, dataset_object,batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]

            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)

            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(vali_data,batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader = self.train_loader
        vali_loader = self.vali_loader
        test_loader = self.test_loader
        # scaler = self.scaler

        train_data=train_loader.dataset
        vali_data=vali_loader.dataset
        test_data=test_loader.dataset

        path = os.path.join(self.args.checkpoints, setting,'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            batch_size = train_loader.batch_size
            num_batches = len(train_loader)
            print(f"Batch size: {batch_size}")
            print(f"Number of batches: {num_batches}")
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(train_data,batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, num_batches, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=False):

        train_loader = self.train_loader
        vali_loader = self.vali_loader
        test_loader = self.test_loader
        scaler = self.scaler

        train_data=train_loader.dataset
        vali_data=vali_loader.dataset
        test_data=test_loader.dataset

        path = os.path.join(self.args.checkpoints, setting)

        test_data=test_loader.dataset

        if test:
            path = os.path.join(self.args.checkpoints, setting,'checkpoints')
            best_model_path = path + '/' + 'checkpoint.pth'

            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = path + '/' +'data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(test_data,batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    fig,axs = plt.subplots(input.shape[-1],1,figsize=(10,2*input.shape[-1]))

                    for j in np.arange(input.shape[-1]) :

                        gt = np.concatenate((input[0, :, j], true[0, :, j]), axis=0)
                        pd = np.concatenate((input[0, :, j], pred[0, :, j]), axis=0)
                        axs[j].plot(gt,label='GroundTruth', linewidth=2)
                        if preds is not None:
                            axs[j].plot(pd, label='Prediction', linewidth=2)                     
                        axs[j].legend()
                    name = os.path.join(folder_path, str(i) + '_' + str(j) + '.pdf')
                    plt.savefig(name, bbox_inches='tight')
        # average_mse = np.average(mses)
        # average_mae = np.average(maes)
        # print('mse:{}, mae:{}'.format(average_mse, average_mae))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = path + '/' +'data'+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        name=str(setting)
        name=str(name.split('_')[-2]+'_'+name.split('_')[-1])
        # get time imformation for saving
        currentTime =gettime()
        with open('new_result.csv', 'a') as file:
            if not os.path.exists('new_result.csv'):
                os.makedirs('new_result.csv')
            
            file.write(f'{currentTime},Model,{self.args.model},seq_len,{self.args.seq_len},label_len,{self.args.label_len},pred_len,{self.args.pred_len}, MSE,{mse}, MAE,{mae},Testname,{name}')
            file.write('\n')
        print("Data written to 'new_result.csv'")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # save the scaler in the same folder
        pickle.dump(scaler, open(folder_path + 'scaler.pkl', 'wb'))
        return

    def predict(self, setting, load=False):
        pred_loader = self.vali_loader
        scaler = self.scaler
        pred_data=pred_loader.dataset
        # if load:
        #     path = os.path.join(self.args.checkpoints, setting,'checkpoints')

        #     #path = os.path.join(self.args.checkpoints, setting)
        #     best_model_path = path + '/' + 'checkpoint.pth'
        #     print(best_model_path)
        #     self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(pred_loader,batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy() )
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('val shape: preds.shape = %s, trues.shape = %s', preds.shape, trues.shape)

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = preds.reshape(-1, 10)

        preds = scaler.inverse_transform(preds)
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues = trues.reshape(-1, 10)

        trues = scaler.inverse_transform(trues)
        path = os.path.join(self.args.checkpoints, setting)
        folder_path = path + '/' +'data'+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        np.save(folder_path + 'real_prediction.npy', preds)

        return
