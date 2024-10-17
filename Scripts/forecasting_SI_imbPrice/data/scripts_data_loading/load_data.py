import math
import os
import numpy as np
import sys
import torch
from datetime import datetime



current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(current_dir,'..','scripts_preprocessing')
sys.path.insert(0,data_processing_dir)
import preprocess_data_DA as pre
import preprocess_data_IMB as pre_IMB



def load_data_DA(days_train=1000,last_ex_test=365,dev='cpu'):

    # Data split
    train_share = 1
    days_train = math.floor(days_train / train_share)  # 64
    last_ex_test = last_ex_test  # 59
    repitition = 1
    la = 24
    lb = 24
    dev = dev
    #loc = "../data/processed_data/SPO_DA/X_df_ds.csv"
    loc = os.path.join(current_dir,'..','processed_data','SPO_DA','X_df_ds.csv')

    data_dict = {
        # Dict containing all info required to retrieve and handle data
        #'loc_data': '../data/processed_data/SPO_DA/',
        'loc_data': loc,
        'feat_cols': ['LOAD_FC', 'GEN_FC', 'FR', 'NL', 'week_day', 'hour-of-day'],
        'lab_cols': ['y'],
        'lookahead': la,
        'days_train': days_train,
        'last_ex_test': last_ex_test,
        'train_share': train_share,
        'val_split_mode': 'alt_test',
        # 'separate' for the validation set right before test set, 'alernating' for train/val examples alternating or 'alt_test' for val/test examples alternating
        'scale_mode': 'norm',  # 'norm','stand' or 'none'
        'scale_base': 'y_hat',  # False or name of column; best to use in combination with scale_mode = 'stand'
        'cols_no_centering': ['y_hat'],
        'scale_price': True,
        'features_mu_calculator': ['price'],
        'labels_mu_calculator': ['mu']
    }



    features, labels = pre.get_data_csv(data_dict, False, True)

    data_np = {
        'train': ([f for f in features[0]],[l for l in labels[0]]),
        'val': ([f for f in features[1]],[l for l in labels[1]]),
        'test': ([f for f in features[2]],[l for l in labels[2]]),
    }


    data_torch = {
        'train': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[0]],
                  [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[0]]),
        'val': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[1]],
                [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[1]]),
        'test': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[2]],
                 [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[2]]),
    }

    return data_np,data_torch



def load_data_MPC(la,lb,dev,data_dict,limit_train_set=None):

    def sample_random_instances(list_arrays, n):

        reduced_arrays = []

        assert list_arrays[0].shape[0] > n, "Size along the zeroth dimension must be larger than n"

        # Set random seed for reproducibility
        np.random.seed(73)

        # Generate random indices
        random_indices = np.random.choice(list_arrays[0].shape[0], n, replace=False)
        random_indices = np.sort(random_indices)

        for array in list_arrays:
            assert array.shape[0] == list_arrays[0].shape[0], "Size of all arrays among zeroth dimension must be the same"
            # Select instances using the random indices
            new_array = array[random_indices]
            reduced_arrays.append(new_array)

        return reduced_arrays,random_indices

    features, labels = pre_IMB.get_data_IMB(data_dict)

    if limit_train_set is not None:
        reduced_arrays,indices = sample_random_instances([features[0][0],features[0][1],labels[0][0]],limit_train_set)
        features[0][0] = reduced_arrays[0]
        features[0][1] = reduced_arrays[1]
        labels[0][0] = reduced_arrays[2]

    data_np = {
        'train': ([f for f in features[0]],[l for l in labels[0]]),
        'val': ([f for f in features[1]],[l for l in labels[1]]),
        'test': ([f for f in features[2]],[l for l in labels[2]]),
    }


    data_torch = {
        'train': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[0]],
                  [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[0]]),
        'val': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[1]],
                [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[1]]),
        'test': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in features[2]],
                 [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in labels[2]]),
    }

    return data_np,data_torch