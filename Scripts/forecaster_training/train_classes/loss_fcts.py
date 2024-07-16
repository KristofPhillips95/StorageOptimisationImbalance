import torch
from torch.utils.data import Dataset


class Loss_Var(torch.nn.Module):

    def __init__(self,rel_weight):
        super(Loss_Var, self).__init__()
        self.rel_weight = rel_weight

    def forward(self,y_hat,y):
        y = y[0]
        y_hat = y_hat[0]

        diff = y-y_hat
        var_y = y[:,1:] - y[:,:-1]
        var_y_hat = y_hat[:,1:] - y_hat[:,:-1]
        diff_var = var_y-var_y_hat

        return torch.sum(torch.square(diff)) + self.rel_weight * torch.sum(torch.square(diff_var))

class Loss_abs_rel(torch.nn.Module):
    def __init__(self,epsilon=1e-2):
        super(Loss_abs_rel,self).__init__()
        self.epsilon = epsilon

    def forward(self,y_hat,y):
        y = y[0]
        y_hat = y_hat[0]

        diff = y-y_hat
        denominator = torch.where(y == 0, torch.abs(y) + self.epsilon, y)

        abs_rel_diff = torch.abs(diff/denominator)

        return torch.mean(abs_rel_diff)

class Loss_Smoothed_Schedule(torch.nn.Module):

    def __init__(self,rel_weight):
        super(Loss_Smoothed_Schedule, self).__init__()
        self.rel_weight = rel_weight
        self.loss_fct_mu = Loss_Var(5)

    def forward(self,y_hat,y):

        mu_hat = y_hat[0]
        d_hat = y_hat[1]
        c_hat = y_hat[2]

        mu = y[0]
        d = y[1]
        c = y[2]

        loss_mu = self.loss_fct_mu([mu_hat],[mu])


        return loss_mu + self.rel_weight*(torch.sum(torch.square(c-c_hat))+torch.sum(torch.square(d-d_hat)))


class Loss_Schedule(torch.nn.Module):

    def __init__(self,eff,power,sm_val):
        super(Loss_Schedule, self).__init__()
        self.eff = eff
        self.power = power
        self.sm_val = sm_val


    def forward(self,y_hat,y,price):

        mu_hat = y_hat[0]

        d = y[1]
        c = y[2]

        d_hat = torch.zeros_like(mu_hat,requires_grad=True)
        c_hat = torch.zeros_like(mu_hat,requires_grad=True)

        mask_charge = price < self.eff * (mu_hat-self.sm_val*self.power)
        mask_discharge = price > (mu_hat+self.sm_val*self.power)/self.eff

        mask_charge_intermediate = (price < self.eff * (mu_hat-self.sm_val*self.power)) & (price < self.eff*mu_hat)
        mask_discharge_intermediate = (price < (mu_hat+self.sm_val*self.power)/self.eff) & (price > mu_hat/self.eff)

        d_hat = torch.where(mask_discharge_intermediate, (self.eff*price-mu_hat)/self.sm_val,d_hat)
        c_hat = torch.where(mask_charge_intermediate,(mu_hat-price/self.eff)/self.sm_val,c_hat)

        d_hat = torch.where(mask_discharge, d_hat + self.power, d_hat)
        c_hat = torch.where(mask_charge, c_hat + self.power, c_hat)

        return torch.mean(torch.square(c-c_hat)) + torch.mean(torch.square(d-d_hat))


#Dataset
class Dataset_Lists(Dataset):
    def __init__(self, feature_tensors, label_tensors):
        self.feature_tensors = feature_tensors
        self.label_tensors = label_tensors
        self.length = self._get_min_length()

    def _get_min_length(self):
        # Assuming all tensors in each list are of the same length
        feature_lengths = [len(t) for t in self.feature_tensors]
        label_lengths = [len(t) for t in self.label_tensors]
        return min(min(feature_lengths), min(label_lengths))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sampled_features = [tensor[idx] for tensor in self.feature_tensors]
        sampled_labels = [tensor[idx] for tensor in self.label_tensors]
        return sampled_features, sampled_labels