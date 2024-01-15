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


#Dataset
class Dataset_Lists(Dataset):
    def __init__(self, feature_tensors, label_tensors=None):
        self.feature_tensors = feature_tensors
        self.label_tensors = label_tensors
        self.length = self._get_min_length()

    def _get_min_length(self):
        # Assuming all tensors in each list are of the same length
        feature_lengths = [len(t) for t in self.feature_tensors]
        if self.label_tensors is None:
            return feature_lengths
        else:
            label_lengths = [len(t) for t in self.label_tensors]
            return min(min(feature_lengths), min(label_lengths))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sampled_features = [tensor[idx] for tensor in self.feature_tensors]
        if self.label_tensors is None:
            return sampled_features
        else:
            sampled_labels = [tensor[idx] for tensor in self.label_tensors]
            return sampled_features, sampled_labels

