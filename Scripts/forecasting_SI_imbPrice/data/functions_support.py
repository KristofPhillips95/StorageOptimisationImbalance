##### LOSS FUNCTIONS AND NN ARCHITECTURES #####

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable,Function
import torch.nn.functional as F
#from sklearn.preprocessing import StandardScaler
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import h5py
import csv
import copy
import sys
import pandas as pd
import datetime as dt



##### OPTIMIZATION MODEL FUNCTIONS #####

def get_opti_matrices(D_out):
    A_latest = np.zeros((D_out, D_out))
    for i in range(D_out):
        for j in range(D_out):
            if i == j:
                if i > 0:
                    A_latest[i, j] = 1
    A_first = np.zeros((D_out, D_out))
    A_last = np.zeros((D_out, D_out))
    A_last[D_out - 1, D_out - 1] = 1
    A_first[0, 0] = 1
    a_first = np.zeros(D_out)
    a_last = np.zeros(D_out)
    a_first[0] = 1
    a_last[D_out - 1] = 1
    A_shift = np.zeros((D_out, D_out))
    for i in range(D_out):
        for j in range(D_out):
            if i == j + 1:
                if i > 0:
                    A_shift[i, j] = 1

    return A_first, A_last, A_latest, A_shift, a_first, a_last

def get_breakpoints(params_dict):
    nb_degr_levels = params_dict['nb_degr_levels']
    max_soc = params_dict['max_soc']
    min_soc = params_dict['min_soc']

    degr_exp = params_dict['degr_exp']
    degr_fact = params_dict['degr_fact']

    soc_breakpoints = np.zeros(nb_degr_levels + 1)
    degr_breakpoints = np.zeros(nb_degr_levels + 1)

    for i in range(nb_degr_levels + 1):
        soc_breakpoints[i] = min_soc + (max_soc - min_soc) / nb_degr_levels * i
        degr_breakpoints[i] = degr_fact * (1 - i / nb_degr_levels) ** (degr_exp)

    return soc_breakpoints, degr_breakpoints

def get_broadcasted_soc_breakpoints(soc_breakpoints, degr_breakpoints, params_dict):
    D_out = params_dict['lookahead']

    bc_soc_bp = np.zeros((D_out, soc_breakpoints.shape[0]))
    bc_degr_bp = np.zeros((D_out, soc_breakpoints.shape[0]))
    bc_fract = np.zeros((D_out, soc_breakpoints.shape[0] - 1))

    for la in range(D_out):
        for bp in range(soc_breakpoints.shape[0]):
            bc_soc_bp[la, bp] = soc_breakpoints[bp]
            bc_degr_bp[la, bp] = degr_breakpoints[bp]
            if bp < soc_breakpoints.shape[0] - 1:
                bc_fract[la, bp] = (degr_breakpoints[bp + 1] - degr_breakpoints[bp]) / (
                            soc_breakpoints[bp + 1] - soc_breakpoints[bp])

    return bc_soc_bp, bc_degr_bp, bc_fract

def interpolate_cds(soc, soc_breakpoints, degr_breakpoints):
    nb_levels = soc_breakpoints.shape[0] - 1
    epsilon = 1e-5

    for lvl in range(nb_levels):
        if soc > soc_breakpoints[lvl + 1] + epsilon:
            pass
        else:
            cds = degr_breakpoints[lvl] + (degr_breakpoints[lvl + 1] - degr_breakpoints[lvl]) / (
            (soc_breakpoints[lvl + 1] - soc_breakpoints[lvl])) * (soc - soc_breakpoints[lvl])
            break

    if 'cds' not in locals():
        print(f"SoC: {soc};SoC breakpoints: {soc_breakpoints}")
        cds = degr_breakpoints[nb_levels]

    return cds

def get_training_labels(train_lab,val_lab,test_lab,decomp='none'):

    if decomp == 'none':
        train_lab_training = train_lab
        val_lab_training = val_lab
        test_lab_training = test_lab
    else:
        x=1#Add code for decomposition

    return train_lab_training,val_lab_training,test_lab_training

def extend_price(price,params_dict,eff=True):

    lookahead = price.shape[0]

    if eff:
        eff_d = params_dict['eff_d']
        eff_c = params_dict['eff_c']
    else:
        eff_d = 1
        eff_c = 1

    extended_price = np.zeros(3*lookahead+1)
    for la in range(lookahead):
        extended_price[la] = price[la]*eff_d
        extended_price[lookahead+la] = -price[la]/eff_c

    return extended_price

def get_profit(net,input_parameters,model_type,input_features,input_labels,decomp):

    optimal_schedule = get_optimal_schedule(net,input_parameters,input_features)

    price_act_ext = np.zeros((input_labels.shape[0], input_labels.shape[1] * 3 + 1))
    for i in range(input_labels.shape[0]):
        price_act_ext[i, :] = extend_price(input_labels[i, :],input_parameters,eff=True)

    profit_per_timestep = np.multiply(price_act_ext,optimal_schedule)

    return np.sum(profit_per_timestep)

def get_profit_prices(input_parameters,price_fc,price_act):

    price_act_ext = np.zeros((price_act.shape[0], price_act.shape[1] * 3 + 1))
    for i in range(price_act.shape[0]):
        price_act_ext[i, :] = extend_price(price_act[i, :],input_parameters,eff=True)

    optimal_schedule = calc_optimal_schedule(price_fc, input_parameters)
    profit_per_timestep = np.multiply(price_act_ext,optimal_schedule)

    return np.sum(profit_per_timestep)

def calc_optimal_schedule(prices,OP_params_dict,extended_price=False,mode='matrix'):
    n_examples = prices.shape[0]
    lookahead = OP_params_dict['lookahead']
    optimal_schedule = np.zeros((n_examples,3*lookahead+1))

    for ex in range(n_examples):
        #print(ex)
        if extended_price or mode == 'explicit':
            price = prices[ex,:]
        else:
            price = extend_price(prices[ex,:],OP_params_dict,eff = True)
        if mode == 'matrix':
            prob,x = opti_da_cvxpy(OP_params_dict,price)
        elif mode == 'explicit':
            prob,x,d = opti_problem_da_explicit(OP_params_dict,price)


        prob.solve(solver='GUROBI', verbose=False)
        #results = opt.solve(prob, tee=False)
        optimal_schedule[ex,:] = x.value

    return optimal_schedule

def opti_da_cvxpy(params_dict, prices):

    A,b = get_full_matrix_problem_DA(params_dict)


    lookahead = params_dict['lookahead']


    x = cp.Variable(lookahead*3+1)

    objective = cp.Maximize(prices@x)
    constraints=[A@x>=b]

    prob=cp.Problem(objective=objective,constraints=constraints)

    return prob,x

def get_optimal_schedule(net,input_parameters,input_features):

    price_fc = input_features @ np.transpose(net)
    optimal_schedule = calc_optimal_schedule(price_fc[:, 0:24]/input_parameters['eff_d'], input_parameters)

    return optimal_schedule

def get_full_matrix_problem_DA(params_dict):
    #Current definition assumes cyclic boundary conditions
    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']
    int_cyclic_bc = int(params_dict['cyclic_bc'])



    A = np.zeros((7 * D_out + 4+2*int_cyclic_bc, 3 * D_out + 1))
    b = np.zeros(7 * D_out + 4+2*int_cyclic_bc)

    start_d = 0
    start_c = D_out
    start_soc = 2 * D_out

    # positivity constraints
    for t in range(D_out):
        A[t + start_d, t + start_d] = 1
        A[t + start_c, t + start_c] = 1
        A[t + start_soc, t + start_soc] = 1
    A[t + start_soc + 1, t + start_soc + 1] = 1

    # Constrain max power
    start_constrain_max_power = 3 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_max_power, t + start_d] = -1
        A[t + start_constrain_max_power, t + start_c] = -1
        b[t + start_constrain_max_power] = -max_charge

    # Constrain max soc
    start_constrain_soc = 4 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_soc, t + start_soc] = -1
        b[t + start_constrain_soc] = -soc_max
    A[t + start_constrain_soc + 1, t + start_soc + 1] = -1
    b[t + start_constrain_soc + 1] = -soc_max

    # SoC update
    start_soc_update_pos = 5 * D_out + 2
    start_soc_update_neg = 6 * D_out + 3
    for t in range(D_out):
        if t == 0:
            A[t + start_soc_update_pos, t + start_soc] = 1
            b[t + start_soc_update_pos] = soc_0

            A[t + start_soc_update_neg, t + start_soc] = -1
            b[t + start_soc_update_neg] = -soc_0

        else:
            A[t + start_soc_update_pos, t + start_soc] = 1
            A[t + start_soc_update_pos, t - 1 + start_soc] = -1
            A[t + start_soc_update_pos, t - 1 + start_d] = 1#/eff_d
            A[t + start_soc_update_pos, t - 1 + start_c] = -1#*eff_c

            A[t + start_soc_update_neg, t + start_soc] = -1
            A[t + start_soc_update_neg, t - 1 + start_soc] = 1
            A[t + start_soc_update_neg, t - 1 + start_d] = -1#/eff_d
            A[t + start_soc_update_neg, t - 1 + start_c] = 1#*eff_c

        A[t + start_soc_update_pos + 1, t + start_soc + 1] = 1
        A[t + start_soc_update_pos + 1, t - 1 + start_soc + 1] = -1
        A[t + start_soc_update_pos + 1, t - 1 + start_d + 1] = 1#/eff_d
        A[t + start_soc_update_pos + 1, t - 1 + start_c + 1] = -1#*eff_c

        A[t + start_soc_update_neg + 1, t + start_soc + 1] = -1
        A[t + start_soc_update_neg + 1, t - 1 + start_soc + 1] = 1
        A[t + start_soc_update_neg + 1, t - 1 + start_d + 1] = -1#/eff_d
        A[t + start_soc_update_neg + 1, t - 1 + start_c + 1] = 1#*eff_c

    if params_dict['cyclic_bc']:

        A[t+start_soc_update_neg + 2, t + start_soc+1] = 1
        b[t + start_soc_update_neg + 2] = soc_0

        A[t+start_soc_update_neg + 3, t + start_soc+1] = -1
        b[t + start_soc_update_neg + 3] = -soc_0


    return A, b

def opti_problem_da_explicit(params_dict,price_fc):



    lookahead = params_dict['lookahead']
    eff = params_dict['eff_d']
    min_soc = params_dict['min_soc']
    max_soc = params_dict['max_soc']
    cyclic_bc = params_dict['cyclic_bc']
    soc_0 = params_dict['soc_0']
    max_charge=params_dict['max_charge']




    d = cp.Variable(lookahead)
    c = cp.Variable(lookahead)
    soc = cp.Variable(lookahead+1)
    variables_combined = cp.hstack([d,c,soc])


    constraints=[]

    constraints.append(soc[0] == soc_0)

    for la in range(lookahead):
        constraints.append(c[la] >= 0)
        constraints.append(d[la] >= 0)
        if params_dict['combined_c_d_max']:
            constraints.append(c[la]+d[la] <= max_charge)
        else:
            constraints.append(c[la] <= max_charge)
            constraints.append(d[la] <= max_charge)
        constraints.append(soc[la] >= min_soc)
        constraints.append(soc[la] <= max_soc)
        constraints.append(soc[la+1] == soc[la] + c[la] - d[la])

    constraints.append(soc[lookahead] >= min_soc)
    constraints.append(soc[lookahead] <= max_soc)


    if cyclic_bc:
        constraints.append(soc[lookahead] == soc_0)

    obj = cp.Maximize(price_fc@(d*eff-c/eff))

    prob = cp.Problem(objective=obj,constraints=constraints)


    return prob,variables_combined,d

def opti_problem(params_dict):
    # Retrieve optimization parameters
    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']
    gamma = params_dict['gamma']
    cyclic_bc = params_dict['cyclic_bc']

    degradation = params_dict['degradation']
    inv_cost = params_dict['inv_cost']

    # Construct matrices to define optimization problem
    A_first, A_last, A_latest, A_shift, a_first, a_last = get_opti_matrices(D_out)

    e_d = cp.Variable(D_out)
    e_c = cp.Variable(D_out)
    soc = cp.Variable(D_out)
    degr_cost = cp.Variable(D_out)

    net_discharge = cp.Variable(D_out)
    price = cp.Parameter(D_in)

    constraints = [e_d >= 0,
                   e_c >= 0,
                   soc >= soc_min,
                   soc <= soc_max,
                   e_d + e_c <= max_charge,
                   net_discharge == e_d * eff_d - e_c / eff_c,
                   A_latest @ soc == A_shift @ soc + A_latest @ (e_c - e_d),
                   A_first @ soc == soc_0 * a_first + A_first @ (e_c - e_d)]

    if degradation:

        degr_levels = params_dict['nb_degr_levels']

        soc_breakpoints, degr_breakpoints = get_breakpoints(params_dict)
        bc_soc_bps, bc_degr_bps, bc_fract = get_broadcasted_soc_breakpoints(soc_breakpoints, degr_breakpoints,
                                                                            params_dict)
        cds_0 = interpolate_cds(soc_0, soc_breakpoints, degr_breakpoints)

        degr_cost = cp.Variable(D_out)
        cds = cp.Variable(D_out)
        soc_per_level = cp.Variable(shape=(D_out, degr_levels))
        degr_level_bool = cp.Variable(shape=(D_out, degr_levels), boolean=True)

        constraints.append(degr_cost >= 0)
        constraints.append(A_latest @ degr_cost >= inv_cost * soc_max * (A_latest @ cds - A_shift @ cds))
        constraints.append(A_first @ degr_cost >= inv_cost * soc_max * (A_first @ cds - cds_0 * a_first))

        constraints.append(cds == cp.sum(
            expr=cp.multiply(degr_level_bool, bc_degr_bps[:, :-1]) + cp.multiply(bc_fract, soc_per_level - cp.multiply(
                degr_level_bool, bc_soc_bps[:, :-1])), axis=1)
                           )

        constraints.append(cp.sum(degr_level_bool, axis=1) == 1)
        constraints.append(cp.sum(soc_per_level, axis=1) == soc)

        constraints.append(cp.multiply(degr_level_bool, bc_soc_bps[:, :-1]) <= soc_per_level)
        constraints.append(cp.multiply(degr_level_bool, bc_soc_bps[:, 1:]) >= soc_per_level)

        objective = cp.Minimize(-cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c) - degr_cost))


    elif gamma != 0:
        constraints.append(degr_cost == 0)
        id = np.identity(D_out)
        #objective = cp.Minimize(
        #    -cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c)) + gamma * cp.quad_form(e_d * eff_d - e_c / eff_c,id))

        #objective = cp.Minimize(-price @ net_discharge + gamma * cp.quad_form(net_discharge,id)) #old
        #objective = cp.Minimize(-price @ net_discharge + gamma * cp.norm(net_discharge,2))

        if params_dict['smoothing'] == 'log-barrier':

            log_barrier = 0

            # Loop through all constraints and add log-barrier for inequalities
            for constr in constraints:
                if isinstance(constr,cp.constraints.Inequality):
                    log_barrier -= gamma * cp.sum(cp.log(-constr.expr))

            objective = cp.Minimize(-price @ net_discharge + log_barrier)

        elif params_dict['smoothing'] == 'quadratic':

            #objective = cp.Minimize(-price @ net_discharge + gamma * cp.norm(net_discharge,2))
            ##### TODO cp.norm(.,2) does not seem to give expected results, check #####
            #objective = cp.Minimize(-price @ net_discharge + gamma * cp.quad_form(net_discharge,id)) #old
            objective = cp.Minimize(-price @ net_discharge + gamma * cp.sum_squares(net_discharge)) #old


        else:
            sys.exit('Invalid smoothing term')

    else:

        constraints.append(degr_cost == 0)
        objective = cp.Minimize(-cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c)))

    if cyclic_bc:
        constraints.append(A_last@soc == soc_0 * a_last)


    prob = cp.Problem(objective=objective, constraints=constraints)

    return prob, [price], [net_discharge,e_d,e_c,soc,degr_cost]

def opti_problem_mu(params_dict):
    """
    Functions does basically the same as opti_problem, but is designed to be exactly the same as what is done in the research proposal
    :param params_dict:
    :return:
    """
    # Retrieve optimization parameters
    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = cp.Parameter()
    soc_0.value = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']
    cyclic_bc = params_dict['cyclic_bc']
    gamma = params_dict['gamma']
    sm = params_dict['smoothing']


    # Construct matrices to define optimization problem
    A_first, A_last, A_latest, A_shift, a_first, a_last = get_opti_matrices(D_out)

    d = cp.Variable(D_out)
    c = cp.Variable(D_out)
    s = cp.Variable(D_out)
    net_discharge = cp.Variable(D_out)

    price = cp.Parameter(D_in)

    constraints = [d >= 0,
                   c >= 0,
                   s >= soc_min,
                   s <= soc_max,
                   d <= max_charge,
                   c <= max_charge,
                   A_first @ s == soc_0 * a_first + A_first @ (c - d),
                   A_latest @ s == A_shift @ s + A_latest @ (c - d),
                   net_discharge == d*eff_d - c/eff_c
                   ]

    #objective = cp.Minimize(-cp.sum(cp.multiply(price, d * eff_d - c / eff_c)+gamma*cp.multiply(d,d)+gamma*cp.multiply(c,c)))
    if gamma == 0:
        objective = cp.Minimize(-price @ (d*eff_d-c/eff_c))  # old
    else:
        if sm == "quadratic":
            objective = cp.Minimize(-price @ (d*eff_d-c/eff_c) + gamma/2 * cp.sum_squares(d) + gamma/2*cp.sum_squares(c))  # old
        elif sm == "logBar":
            objective = cp.Minimize(-price @ (d*eff_d-c/eff_c) - gamma * (
                                    cp.sum(cp.log(d)) +
                                    cp.sum(cp.log(max_charge-d)) +
                                    cp.sum(cp.log(c)) +
                                    cp.sum(cp.log(max_charge - c))
                                    ))

        #TODO: adjust code to also include the possibility of a joint charge/discharge limit in log-barrier

    if cyclic_bc:
        constraints.append(A_last@s == soc_0 * a_last)


    prob = cp.Problem(objective=objective, constraints=constraints)

    return prob, [price,soc_0], [net_discharge,d,c,s]



##### DATA PRE-PROCESSING #####


#Data reading, feature/label split, forward transformation

def get_indices(last_ex_test,days_train,train_share,size_array,mode):

    tot_n_examples = last_ex_test + days_train

    idx_test = [-(i+1) for i in range(last_ex_test)]

    start = size_array-tot_n_examples

    if mode == 'separate':
        idx_train = [i for i in range(round(days_train*train_share))]
        idx_val = [round(days_train*train_share) + i for i in range(days_train - round(days_train*train_share))]

    elif mode == 'alternating':

        idx_train = [0]
        idx_val = [1]

        for i in range(days_train-2):
            share = len(idx_train)/(len(idx_train)+len(idx_val))

            if share < train_share:
                idx_train.append(i+2)
            else:
                idx_val.append(i+2)

    elif mode == 'alt_test':

        idx_train = [i for i in range(int(round(days_train*train_share)))]
        idx_val = [int(round(days_train * train_share))+1 + 2 * (i - 1) + 1 for i in range(int(round(last_ex_test / 2)))]
        idx_test = [int(round(days_train*train_share))+1+2*i for i in range(int(round(last_ex_test/2)))]

        idx_test = [idx + start for idx in idx_test]

    idx_train = [idx + start for idx in idx_train]
    idx_val = [idx + start for idx in idx_val]


    return idx_train,idx_val,idx_test

def preprocess_data(data_dict):
    #Retrieve relevant parameters from dictionary
    loc_data = data_dict['loc_data']
    scale_mode = data_dict['scale_mode']
    scale_base = data_dict['scale_base']
    cols_features = data_dict['feat_cols']
    col_label_price = data_dict['col_label_price']
    col_label_fc_price = data_dict['col_label_fc_price']
    lookahead = data_dict['lookahead']
    days_train = data_dict['days_train']
    last_ex_test = data_dict['last_ex_test']
    train_share = data_dict['train_share']
    cols_no_centering = data_dict['cols_no_centering']

    #Read data in dataframe and scale
    data_all = get_data_pandas(loc_data=loc_data)
    #data_all_scaled = scale_df(df=data_all,mode=scale_mode,base=scale_base)
    data_all_scaled,stdev_yhat = scale_df_new(data_all,scale_mode,cols_no_centering)
    data_dict["stdev_yhat"] = stdev_yhat
    #Get features and prices in numpy
    features = data_all[cols_features].to_numpy()
    scaled_features = data_all_scaled[cols_features].to_numpy()
    prices = data_all[col_label_price].to_numpy()
    forecasted_prices = data_all[col_label_fc_price].to_numpy()
    forecasted_prices_scaled = data_all_scaled[col_label_fc_price].to_numpy()

    #Select scaled or unscaled features
    if scale_mode == 'none':
        features_for_training = features
    else:
        features_for_training = scaled_features

    #Re-organize features and labels according to lookahead horizon
    forward_features, forward_prices = get_forward_features_labels(features_for_training, prices, lookahead)
    _, forward_fc_price = get_forward_features_labels(features_for_training, forecasted_prices, lookahead)
    _, forward_fc_price_scaled = get_forward_features_labels(features_for_training, forecasted_prices_scaled, lookahead)

    indices_train,indices_val,indices_test = get_indices(last_ex_test,days_train,train_share,forward_features.shape[0],data_dict['val_split_mode'])

    features_train = forward_features[indices_train,:]
    features_val = forward_features[indices_val,:]
    features_test = forward_features[indices_test,:]

    price_train = forward_prices[indices_train,:]
    price_val = forward_prices[indices_val,:]
    price_test = forward_prices[indices_test,:]

    price_fc_train = forward_fc_price[indices_train,:]
    price_fc_val = forward_fc_price[indices_val,:]
    price_fc_test = forward_fc_price[indices_test,:]

    price_fc_train_scaled = forward_fc_price_scaled[indices_train, :]
    price_fc_val_scaled = forward_fc_price_scaled[indices_val, :]
    price_fc_test_scaled = forward_fc_price_scaled[indices_test, :]

    return [features_train], [features_val], [features_test], price_train, price_val, price_test, [price_fc_train,price_fc_val, price_fc_test], [price_fc_train_scaled,price_fc_val_scaled,price_fc_test_scaled]

def get_data_pandas(loc_data):
    data_forecast = pd.read_csv(loc_data + 'forecast_df.csv')
    data = pd.read_csv(loc_data + 'X_df_ds.csv')

    data['dt'] = data['ds'].str[:19]
    # data_other['h'] = pd.to_timedelta(data_other['ds'].str[-4].astype(int),'h')
    # data_other['Datetime'] = pd.to_datetime(data_other['dt'])+pd.DateOffset(hours=data_other['h'])
    data['Datetime'] = pd.to_datetime(data['dt'])  # + data_other['h']
    data['date'] = data['Datetime'].dt.date

    data_forecast['dt'] = data_forecast['ds'].str[:19]
    data_forecast['Datetime'] = pd.to_datetime(data_forecast['dt'])
    data_forecast['date'] = data_forecast['Datetime'].dt.date

    dates_rem = [dt.date(2019, 3, 18), dt.date(2019, 3, 31), dt.date(2020, 3, 29), dt.date(2021, 3, 28),
                 dt.date(2022, 3, 27), dt.date(2023, 3, 26),
                 dt.date(2019, 10, 27), dt.date(2020, 10, 25), dt.date(2021, 10, 31), dt.date(2022, 10, 30)]

    data = data[~(data['date'].isin(dates_rem))][1:]
    data_forecast = data_forecast[~(data_forecast['date'].isin(dates_rem))]

    data_all = pd.merge(left=data_forecast, right=data, on='Datetime', how='inner')

    return data_all

def scale_df(df,mode,base=False):

    df_scaled = copy.deepcopy(df)

    if not isinstance(base, str):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_scaled[col] = scale_column(df[col], mode)
    else:
        stdev_base = df[base].std()
        for col in df.columns:
            if col != base and pd.api.types.is_numeric_dtype(df[col]):
                df_scaled[col] = scale_column(df[col], mode) * stdev_base

    # for col in df.columns:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         df_scaled[col] = scale_column(column=df[col],mode=mode)

    return df_scaled

def scale_df_new(df,mode,cols_no_centering):

    df_scaled = copy.deepcopy(df)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in cols_no_centering:
                df_scaled[col] = scale_column(df[col], "stand_no_centering")
            else:
                df_scaled[col] = scale_column(df[col], mode)

    stdev_yhat = 0
    if "y_hat" in df.columns:
        stdev_yhat = np.std(df["y_hat"])

    return df_scaled,stdev_yhat

def scale_column(column,mode):

    if mode == 'stand':
        scaled_column = (column-column.mean())/column.std()
    elif mode == 'norm':
        scaled_column = (column-column.min())/(column.max()-column.min())
    elif mode == 'none':
        scaled_column = column
    elif mode == 'stand_no_centering':
        scaled_column = column/column.std()
    else:
        sys.exit('invalid scaling mode')
    return scaled_column

def get_forward_features_labels(features,labels,lookahead):
    n_rows = features.shape[0]
    n_features = features.shape[1]
    n_examples = int(n_rows/lookahead)


    forward_features=np.zeros((n_examples,n_features*lookahead))
    forward_labels = np.zeros((n_examples,lookahead))

    for ex in range(n_examples):
        for la in range(lookahead):
            forward_labels[ex,la] = labels[lookahead*ex+la]
            for feat in range(n_features):
                forward_features[ex,n_features*la+feat] = features[lookahead*ex+la,feat]
    return forward_features,forward_labels


#Retrieve correct labels

def preprocess_labels(training_dict, price_train, price_val, price_test, OP_params_dict, data_dict):

    def aggregate_extended_sched(extended_sched):
        la = OP_params_dict['lookahead']
        n_ex = extended_sched.shape[0]
        net_discharge = np.zeros((n_ex,la))

        for i in range(n_ex):
            net_discharge[i,:] = extended_sched[i,0:la]*OP_params_dict['eff_d'] - extended_sched[i,la:2*la]/OP_params_dict['eff_c']

        return net_discharge

    type = training_dict['type_train_labels']
    sched_type = training_dict['sched_type']

    if data_dict["scale_price"]:
        price_train = price_train/data_dict["stdev_yhat"]
        price_val = price_val/data_dict["stdev_yhat"]
        price_test = price_test/data_dict["stdev_yhat"]

    if type == 'price':
        labels_train = price_train
        labels_val = price_val
        labels_test = price_test
    elif type == 'price_schedule':
        sched_train = calc_optimal_schedule(price_train,OP_params_dict,mode='explicit')
        sched_val = calc_optimal_schedule(price_val,OP_params_dict,mode='explicit')
        sched_test = calc_optimal_schedule(price_test,OP_params_dict,mode='explicit')
        #TODO: check impact of factor size: this way probably augments the size twice in new cases of train/val set?
        if sched_type == "net_discharge":
            sched_train = aggregate_extended_sched(sched_train)
            sched_val = aggregate_extended_sched(sched_val)
            sched_test = aggregate_extended_sched(sched_test)
        labels_train = [price_train,sched_train,np.sum(price_train*sched_train,axis=1)]
        labels_val = [price_val,sched_val,np.sum(price_val*sched_val,axis=1)]
        labels_test = [price_test,sched_test,np.sum(price_test*sched_test,axis=1)]

        #labels_val = price_val
        #labels_test = price_test

    return labels_train,labels_val,labels_test

def retrieve_optimal_schedule_train_test(train_prices,validation_prices, OP_params_dict,mode='matrix'):

    #TODO: remove function?

    optimized_train_sched = calc_optimal_schedule(train_prices,OP_params_dict,mode=mode)
    optimized_val_sched = calc_optimal_schedule(validation_prices,OP_params_dict,mode=mode)

    return optimized_train_sched,optimized_val_sched


#Convert to tensor and train dataset
def get_train_validation_tensors(train_features, train_labels, validation_features, validation_labels, test_features, test_labels):

    if isinstance(train_features,list):
        tensor_train_features = []
        tensor_validation_features = []
        tensor_test_features = []
        for labels in train_features:
            tensor_train_features.append(torch.from_numpy(labels).float())
        for labels in validation_features:
            tensor_validation_features.append(torch.from_numpy(labels).float())
        for labels in test_features:
            tensor_test_features.append(torch.from_numpy(labels).float())

    else:
        tensor_train_features = torch.from_numpy(train_features).float()
        tensor_validation_features = torch.from_numpy(validation_features).float()
        tensor_test_features = torch.from_numpy(test_features).float()

    if isinstance(train_labels,list):
        tensor_train_labels = []
        tensor_validation_labels = []
        tensor_test_labels = []
        for labels in train_labels:
            tensor_train_labels.append(torch.from_numpy(labels).float())
        for labels in validation_labels:
            tensor_validation_labels.append(torch.from_numpy(labels).float())
        for labels in test_labels:
            tensor_test_labels.append(torch.from_numpy(labels).float())

        #tensor_train_labels = [torch.from_numpy(train_labels[0]).float(),torch.from_numpy(train_labels[1]).float()]
    else:
        tensor_train_labels = torch.from_numpy(train_labels).float()

    return [tensor_train_features, tensor_train_labels, tensor_validation_features, tensor_validation_labels, tensor_test_features, tensor_test_labels]

def set_tensors_to_device(list_tensors,dev):
    # Set list of tensors to specified device

    global_list = []

    for tensors in list_tensors:
        if type(tensors) is list:
            new_entry = [t.to(dev) for t in tensors]
        else:
            new_entry = tensors.to(dev)
        global_list.append(new_entry)

    return global_list

def get_train_Dataset(model_type, type_train_labels, train_feat, train_lab_training):

    if model_type == 'edRNN':
        if type_train_labels == 'price_schedule':
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training[0],
                                                           train_lab_training[1])
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training)
    else:
        if type_train_labels == 'price_schedule':
            #train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training[0], train_lab_training[1])
            train_Dataset = torch.utils.data.TensorDataset(train_feat, *train_lab_training)
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training)

    return train_Dataset


# Splitting data set in train & validation set

def split_features_labels_train_validation(features, labels, amount_of_validation_periods: int, share_validation,
                                           lookahead: int):
    if amount_of_validation_periods <= 1:
        print('Minimum amount of valdiation periods is 2')
        sys.exit()

    total_instances = labels.shape[0]

    start_array_val, end_array_val, start_array_train, end_array_train = get_pos_arrays(amount_of_validation_periods,
                                                                                        total_instances, lookahead,
                                                                                        share_validation)

    return get_arrays_from_positions(features, labels, start_array_val, end_array_val, start_array_train,
                                     end_array_train, amount_of_validation_periods)

def get_lost_instances(amount_of_validation_periods, lookahead):
    if amount_of_validation_periods < 2:
        return amount_of_validation_periods * lookahead
    elif amount_of_validation_periods >= 2:
        return (amount_of_validation_periods - 1) * 2 * lookahead

def get_pos_arrays(amount_of_validation_periods, total_instances, lookahead, share_validation):
    amount_of_training_periods = amount_of_validation_periods - 1

    leftover_instances = total_instances - get_lost_instances(amount_of_validation_periods, lookahead)

    instances_validation = share_validation * leftover_instances

    instance_per_validation_period = round(instances_validation / amount_of_validation_periods)
    instance_per_training_period = round((leftover_instances - instances_validation) / amount_of_training_periods)

    start_array_val = []
    end_array_val = []
    start_array_train = []
    end_array_train = []

    validation_periods_defined = 0

    while validation_periods_defined < amount_of_validation_periods - 1:
        start = validation_periods_defined * (
                    instance_per_validation_period + 2 * lookahead + instance_per_training_period)

        start_array_val.append(start)
        end_array_val.append(start + instance_per_validation_period)
        start_array_train.append(start + instance_per_validation_period + lookahead)
        end_array_train.append(start + instance_per_validation_period + lookahead + instance_per_training_period)

        validation_periods_defined += 1

    start_array_val.append(
        validation_periods_defined * (instance_per_validation_period + 2 * lookahead + instance_per_training_period))
    end_array_val.append(total_instances)

    return start_array_val, end_array_val, start_array_train, end_array_train

def get_arrays_from_positions(features, labels, start_array_val, end_array_val, start_array_train, end_array_train,
                              amount_of_validation_periods):
    labels_val = labels[start_array_val[0]:end_array_val[0]]
    labels_train = labels[start_array_train[0]:end_array_train[0]]

    if type(features) == list:
        features_past = features[0]
        features_future = features[1]

        features_val_past = features_past[start_array_val[0]:end_array_val[0], :]
        features_train_past = features_past[start_array_train[0]:end_array_train[0], :]
        features_val_fut = features_future[start_array_val[0]:end_array_val[0], :]
        features_train_fut = features_future[start_array_train[0]:end_array_train[0], :]

        periods_determined = 1

        while periods_determined < amount_of_validation_periods:

            if periods_determined < amount_of_validation_periods - 1:
                features_train_past = np.concatenate((features_train_past, features_past[
                                                                           start_array_train[periods_determined]:
                                                                           end_array_train[periods_determined], :]))
                features_train_fut = np.concatenate((features_train_fut, features_future[
                                                                         start_array_train[periods_determined]:
                                                                         end_array_train[periods_determined], :]))
                labels_train = np.concatenate(
                    (labels_train, labels[start_array_train[periods_determined]:end_array_train[periods_determined]]))

            features_val_past = np.concatenate((features_val_past, features_past[
                                                                   start_array_val[periods_determined]:end_array_val[
                                                                       periods_determined], :]))
            features_val_fut = np.concatenate((features_val_fut, features_future[
                                                                 start_array_val[periods_determined]:end_array_val[
                                                                     periods_determined], :]))

            labels_val = np.concatenate(
                (labels_val, labels[start_array_val[periods_determined]:end_array_val[periods_determined]]))

            periods_determined += 1

        features_train = [features_train_past, features_train_fut]
        features_val = [features_val_past, features_val_fut]


    else:
        features_val = features[start_array_val[0]:end_array_val[0], :]
        features_train = features[start_array_train[0]:end_array_train[0], :]

        periods_determined = 1

        while periods_determined < amount_of_validation_periods:

            if periods_determined < amount_of_validation_periods - 1:
                features_train = np.concatenate((features_train, features[
                                                                 start_array_train[periods_determined]:end_array_train[
                                                                     periods_determined], :]))
                labels_train = np.concatenate(
                    (labels_train, labels[start_array_train[periods_determined]:end_array_train[periods_determined]]))

            features_val = np.concatenate(
                (features_val, features[start_array_val[periods_determined]:end_array_val[periods_determined], :]))
            labels_val = np.concatenate(
                (labels_val, labels[start_array_val[periods_determined]:end_array_val[periods_determined]]))

            periods_determined += 1

    return features_train, features_val, labels_train, labels_val

#Save outcome
def save_outcome(list_dict_outcome, store_code):
    # Function saving all information in a list of dicts coming out of the NN training procedure in specified location

    def convert_nn_arch(list_dict_outcome):
        n_hidden_layers = len(list_dict_outcome[0]['list_units'])

        for dict in list_dict_outcome:
            for hl in range(n_hidden_layers):
                dict[f'hp_hl{hl+1}_units'] = dict['list_units'][hl]
                dict[f'hp_hl{hl+1}_act'] = dict['list_act'][hl]

            del dict['list_units']
            del dict['list_act']

        return list_dict_outcome


    for dict_out in list_dict_outcome:
        path = f"{store_code}config_{dict_out['a_config']}.pt"
        torch.save(dict_out['trained_net_opti'], path)
        del dict_out['trained_net_opti']

        path = f"{store_code}config_{dict_out['a_config']}_profitEvol.npz"
        path_h5py = f"{store_code}config_{dict_out['a_config']}_train_evols.h5"
        save_arrays = []
        for loss_str in ['profit', 'mse_sched', 'mse_sched_weighted', 'mse_price']:
            for (i,set) in enumerate(['train', 'val', 'test']):
                save_arrays.append(f"loss_{loss_str}_{set}")
        #save_arrays = ['profit_evol_train_RA','profit_evol_train_RN','profit_evol_val_RA','profit_evol_val_RN','profit_evol_test_RA','profit_evol_test_RN','n_gradients_zero','regret_evol_train', 'regret_evol_val', 'regret_evol_test']

        with h5py.File(path_h5py, 'w') as f:
            for key, value in {k:dict_out[k] for k in save_arrays}.items():
                f.create_dataset(key, data=value)

        for evol in save_arrays:
            del dict_out[evol]


    list_dict_outcome = convert_nn_arch(list_dict_outcome)

    dict_outcome = {}

    for d in list_dict_outcome:
        for key, value in d.items():
            if key not in dict_outcome:
                dict_outcome[key] = []
            dict_outcome[key].append(value)

    path = f"{store_code}outcome.csv"

    with open(path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict_outcome.keys())
        writer.writerows(zip(*dict_outcome.values()))

