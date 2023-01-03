import torch.nn as nn
import numpy as np


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        pass

    def __call__(self, cri_out, net_out, batch):
        # 
        for key in net_out:
            if hasattr(net_out[key], 'cpu'):
                if key == 'y0' or key == 'y1':
                    net_out[key] = net_out[key] * batch['y_std'] + batch['y_mu']
                net_out[key] = net_out[key].data.cpu().numpy()
        for key in batch:
            if hasattr(batch[key], 'cpu'):
                batch[key] = batch[key].data.cpu().numpy()
        
        batch['y'] = batch['y'] * batch['y_std'] + batch['y_mu'] 
        ite = self.rmse_ite(net_out, batch)
        ate = self.abs_ate(net_out, batch)
        pehe = self.pehe(net_out, batch)
        rmse_factual, rmse_cfactual = self.y_errors(net_out, batch)
        out = {}
        out['ite'] = ite
        out['ate'] = ate
        out['pehe'] = pehe
        out['rmse_factual'], out['rmse_cfactual'] = rmse_factual, rmse_cfactual
        return out

    def rmse_ite(self, net_out, batch):
        pred_ite = np.zeros_like(batch['mu_1'] - batch['mu_0'])
        idx1, idx0 = np.where(batch['t'] == 1), np.where(batch['t'] == 0)
        ite1, ite0 = batch['y'][idx1] - net_out['y0'][idx1], net_out['y1'][idx0] - batch['y'][idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(batch['mu_1'] - batch['mu_0'] - pred_ite)))

    def abs_ate(self, net_out, batch):
        return np.abs(np.mean(net_out['y1'] - net_out['y0']) - np.mean(batch['mu_1'] - batch['mu_0']))

    def pehe(self, net_out, batch):
        return np.sqrt(np.mean(np.square((batch['mu_1'] - batch['mu_0']) - (net_out['y1'] - net_out['y0']))))

    def y_errors(self, net_out, batch):
        ypred = (1 - batch['t']) * net_out['y0'] + batch['t'] * net_out['y1']
        ypred_cf = batch['t'] * net_out['y0'] + (1 - batch['t']) * net_out['y1']
        return self.y_errors_pcf(ypred, ypred_cf, batch)

    def y_errors_pcf(self, ypred, ypred_cf, batch):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - batch['y'])))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - batch['y_cf'])))
        return rmse_factual, rmse_cfactual

