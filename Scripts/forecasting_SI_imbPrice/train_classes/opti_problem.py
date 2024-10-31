import numpy as np
import cvxpy as cp
import torch
import torch_classes as ct
import time

class Application():

    def __init__(self,type,OP_params,loss_fct):

        self.la = OP_params['lookahead']
        self.soc_0 = OP_params['soc_0']
        self.OP_params = OP_params
        self.type = type
        self.op = OptiProblemNew(OP_params)
        self.loss = loss_fct

    def __call__(self,fc,lab):
        """
        Function that runs the application for a specified forecast
        :params
        -forecast: n_ex x la np array
            n_ex: number of examples
            la: lookahead
        :return:
        -outcome: n_ex x la np
        """

        n_ex = fc.shape[0]
        fw_net_discharge = np.zeros((n_ex,self.la))
        fw_soc = np.zeros((n_ex,self.la))
        fw_c = np.zeros((n_ex,self.la))
        fw_d = np.zeros((n_ex,self.la))
        fw_profit = np.zeros(n_ex)

        print("Starting optimization")
        tic = time.time()

        for i in range(n_ex):
            if self.type == 'MPC':
                if i > 0:
                    s0 = self.soc_0 + fw_c[i-1,0] - fw_d[i-1,0] #Update State of charge based on previous action
                    s0 = min(s0,self.OP_params['max_soc'])
                    s0 = max(s0,self.OP_params['min_soc'])
                    self.soc_0 = s0
                else:
                    self.soc_0 = self.OP_params['soc_0']

            net_discharge,d,c,soc,mu = self.op([self.soc_0,fc[i,:]])

            fw_net_discharge[i,:] = net_discharge
            fw_d[i,:] = d
            fw_c[i,:] = c
            fw_soc[i,:] = soc

        loss_val = -self.loss([torch.tensor(fw_net_discharge)],[torch.tensor(lab)]).item()

        print(f"Total time for optimization: {time.time()-tic} \t\t Opti time per example: {(time.time()-tic)/n_ex}")

        return fw_net_discharge,fw_soc,loss_val



    def set_loss_fct(self,loss_str):
        if loss_str in ['profit', 'mse_sched', 'mse_sched_weighted','mse_price','mae_price','pinball']:
            self.loss_fct = ct.Loss(loss_str,self.training_params['loss_params'])
        else:
            raise ValueError(f"Loss function {loss_str} not supported")

class OptiProblem():

    def __init__(self,OP_params):

        super(OptiProblem,self).__init__()
        self.OP_params = OP_params

    def get_opti_matrices(self):
        D_out = self.OP_params['lookahead']
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

    def get_opti_problem(self,MPC=False):
        """
        Functions does basically the same as opti_problem, but is designed to be exactly the same as what is done in the research proposal
        :param params_dict:
        :return:
        """
        # Retrieve optimization parameters
        D_out = self.OP_params['lookahead']
        D_in = D_out
        eff_c = self.OP_params['eff_c']
        eff_d = self.OP_params['eff_d']

        soc_max = self.OP_params['max_soc']
        soc_min = self.OP_params['min_soc']
        max_charge = self.OP_params['max_charge']
        cyclic_bc = self.OP_params['cyclic_bc']
        gamma = self.OP_params['gamma']
        sm = self.OP_params['smoothing']

        # Construct matrices to define optimization problem
        A_first, A_last, A_latest, A_shift, a_first, a_last = self.get_opti_matrices()

        d = cp.Variable(D_out)
        c = cp.Variable(D_out)
        s = cp.Variable(D_out)
        net_discharge = cp.Variable(D_out)
        mu = cp.Variable(D_out)

        if MPC:
            soc_0 = cp.Parameter()
            soc_0.value = self.OP_params['soc_0']
        else:
            soc_0 = self.OP_params['soc_0']


        price = cp.Parameter(D_in)

        constraints = [d >= 0,
                       c >= 0,
                       s >= soc_min,
                       s <= soc_max,
                       d <= max_charge,
                       c <= max_charge,
                       A_first @ s == soc_0 * a_first + A_first @ (c - d),
                       A_latest @ s == A_shift @ s + A_latest @ (c - d),
                       net_discharge == d * eff_d - c / eff_c
                       ]
        #
        # constraint_dual = [mu[0] == constraints[6].dual_value[0]]
        #
        # for i in range(D_out-1):
        #     constraint_dual.append(mu[i+1] == constraints[7].dual_value[i+1])
        #
        #


        # objective = cp.Minimize(-cp.sum(cp.multiply(price, d * eff_d - c / eff_c)+gamma*cp.multiply(d,d)+gamma*cp.multiply(c,c)))
        if gamma == 0:
            objective = cp.Minimize(-price @ (d * eff_d - c / eff_c))  # old
        else:
            if sm == "quadratic":
                objective = cp.Minimize(
                    -price @ (d * eff_d - c / eff_c) + gamma / 2 * cp.sum_squares(d) + gamma / 2 * cp.sum_squares(
                        c))  # old
            elif sm == "logBar":
                objective = cp.Minimize(-price @ (d * eff_d - c / eff_c) - gamma * (
                        cp.sum(cp.log(d)) +
                        cp.sum(cp.log(max_charge - d)) +
                        cp.sum(cp.log(c)) +
                        cp.sum(cp.log(max_charge - c))
                ))

            # TODO: adjust code to also include the possibility of a joint charge/discharge limit in log-barrier

        if cyclic_bc:
            constraints.append(A_last @ s == soc_0 * a_last)

        prob = cp.Problem(objective=objective, constraints=constraints)

        if MPC:
            return prob, [price,soc_0], [net_discharge, d, c, s]
        else:
            return prob, [price], [net_discharge, d, c, s]

class OptiProblemNew():
    """
    Class of Optimization problems
    """
    def __init__(self,OP_params,bs=1):

        super(OptiProblemNew,self).__init__()
        self.OP_params = OP_params
        self.bs = bs
        if bs == 1:
            self.init_opti_problem()
        else:
            self.init_opti_problem_batched(bs)

    def __call__(self,param_values):
        """
        Function that:
        (i) sets the variable parameter values to those given as function arguments --> problem is uniquely defined
        (ii) solves the optimization problem
        (iii) retrieves the optimized values of the decision variables

        :params:
        -params_values: list np arrays
            list of arrays containing the values of variable parameters corresponding to those contained in self.params

        :return:
        - var_vals: list of np arrays
            list of arrays containing the optimized values of the decision variables corresponding to those in self.vars
        """

        self.set_params_opti(param_values)

        try:
            #self.prob.solve(solver=cp.GUROBI)
            self.prob.solve(solver=cp.ECOS)
        except Exception as e:
            print("An error occurred:", e)
            print("Swithing to SCS solver")
            self.prob.solve(solver=cp.SCS)


        var_vals = self.get_var_values()

        return var_vals

    def init_opti_problem(self):
        """
        Defines the optimization problem by setting the non-variable parameters.

        :params:
        -params_dict: dict
            Dictionary containing the relevant non-variable parameters required to define the optimization problem

        :set:
        -self.prob: cp.Problem()
            cvxpy problem that can be optimized
        -self.params: list of cp.Parameter
            list of all the variable parameters that need to be defined before calling the optimization
        -self.vars: list of cp.Var
            list of all the variables of the optimization problem

        :return:
            NA
        """
        # Retrieve optimization parameters
        D_out = self.OP_params['lookahead']
        D_in = D_out
        eff_c = self.OP_params['eff_c']
        eff_d = self.OP_params['eff_d']
        soc_max = self.OP_params['max_soc']
        soc_min = self.OP_params['min_soc']
        max_charge = self.OP_params['max_charge']
        cyclic_bc = self.OP_params['cyclic_bc']
        gamma = self.OP_params['gamma']
        sm = self.OP_params['smoothing']

        # Construct matrices to define optimization problem
        A_first, A_last, A_latest, A_shift, a_first, a_last = self.get_opti_matrices()

        d = cp.Variable(D_out)
        c = cp.Variable(D_out)
        s = cp.Variable(D_out)
        net_discharge = cp.Variable(D_out)

        price = cp.Parameter(D_in)
        soc_0 = cp.Parameter()

        constraints = [d >= 0,
                       c >= 0,
                       s >= soc_min,
                       s <= soc_max,
                       d <= max_charge,
                       c <= max_charge,
                       A_first @ s == soc_0 * a_first + A_first @ (c - d),
                       A_latest @ s == A_shift @ s + A_latest @ (c - d),
                       net_discharge == d * eff_d - c / eff_c
                       ]


        # objective = cp.Minimize(-cp.sum(cp.multiply(price, d * eff_d - c / eff_c)+gamma*cp.multiply(d,d)+gamma*cp.multiply(c,c)))
        if gamma == 0:
            objective = cp.Minimize(-price @ (d * eff_d - c / eff_c))  # old
        else:
            if sm in ["quadratic","quadratic_symm","piecewise"]:
                if self.OP_params['include_soc_smoothing']:
                    print("SoC smoothing included")
                    objective = cp.Minimize(
                        -price @ (d * eff_d - c / eff_c) + gamma / 2 * cp.sum_squares(d) + gamma / 2 * cp.sum_squares(
                            c) + gamma / 2 * cp.sum_squares(s) )
                else:
                    print("SoC smoothing NOT included")
                    objective = cp.Minimize(
                        -price @ (d * eff_d - c / eff_c) + gamma / 2 * cp.sum_squares(d) + gamma / 2 * cp.sum_squares(c))
            elif sm in ["logBar","logistic"]:
                if self.OP_params['include_soc_smoothing']:
                    objective = cp.Minimize(-price @ (d * eff_d - c / eff_c) - gamma * (
                            cp.sum(cp.log(d)) +
                            cp.sum(cp.log(max_charge - d)) +
                            cp.sum(cp.log(c)) +
                            cp.sum(cp.log(max_charge - c)) +
                            cp.sum(cp.log(s-soc_min)) +
                            cp.sum(cp.log(soc_max - s))
                    ))
                else:
                    objective = cp.Minimize(-price @ (d * eff_d - c / eff_c) - gamma * (
                            cp.sum(cp.log(d)) +
                            cp.sum(cp.log(max_charge - d)) +
                            cp.sum(cp.log(c)) +
                            cp.sum(cp.log(max_charge - c))
                    ))

            # TODO: adjust code to also include the possibility of a joint charge/discharge limit in log-barrier

        if cyclic_bc:
            constraints.append(A_last @ s == soc_0 * a_last)

        prob = cp.Problem(objective=objective, constraints=constraints)

        self.prob = prob
        self.params = [soc_0,price]
        self.vars = [net_discharge,d,c,s]
        self.constraints = constraints

    def init_opti_problem_batched(self, bs):
        """
        Functions does basically the same as opti_problem, but is designed to be exactly the same as what is done in the research proposal
        :param params_dict:
        :return:
        """
        # Retrieve optimization parameters
        D_out = self.OP_params['lookahead']
        D_in = D_out
        eff_c = self.OP_params['eff_c']
        eff_d = self.OP_params['eff_d']
        soc_0 = self.OP_params['soc_0']
        soc_max = self.OP_params['max_soc']
        soc_min = self.OP_params['min_soc']
        max_charge = self.OP_params['max_charge']
        cyclic_bc = self.OP_params['cyclic_bc']
        gamma = self.OP_params['gamma']
        sm = self.OP_params['smoothing']

        # Construct matrices to define optimization problem
        A_first, A_last, A_latest, A_shift, a_first, a_last = self.get_opti_matrices()

        d = cp.Variable((bs, D_out))
        c = cp.Variable((bs, D_out))
        s = cp.Variable((bs, D_out))
        net_discharge = cp.Variable((bs, D_out))

        price = cp.Parameter((bs, D_in))
        soc_0 = cp.Parameter()

        constraints = [d >= 0,
                       c >= 0,
                       s >= soc_min,
                       s <= soc_max,
                       d <= max_charge,
                       c <= max_charge,
                       net_discharge == d * eff_d - c / eff_c,
                       ]

        for i in range(bs):
            constraints.append(A_first @ s[i] == soc_0 * a_first + A_first @ (c[i] - d[i]))
            constraints.append(A_latest @ s[i] == A_shift @ s[i] + A_latest @ (c[i] - d[i]))
        #
        # constraint_dual = [mu[0] == constraints[6].dual_value[0]]
        #
        # for i in range(D_out-1):
        #     constraint_dual.append(mu[i+1] == constraints[7].dual_value[i+1])
        #
        #

        # objective = cp.Minimize(-cp.sum(cp.multiply(price, d * eff_d - c / eff_c)+gamma*cp.multiply(d,d)+gamma*cp.multiply(c,c)))
        if gamma == 0:
            objective = cp.Minimize(cp.sum([-price[i] @ (d[i] * eff_d - c[i] / eff_c) for i in range(bs)]))  # old

        else:
            if sm == "quadratic":
                objective = cp.Minimize(
                    -price @ (d * eff_d - c / eff_c) + gamma / 2 * cp.sum_squares(d) + gamma / 2 * cp.sum_squares(
                        c))  # old
            elif sm == "logBar":
                objective = cp.Minimize(-price @ (d * eff_d - c / eff_c) - gamma * (
                        cp.sum(cp.log(d)) +
                        cp.sum(cp.log(max_charge - d)) +
                        cp.sum(cp.log(c)) +
                        cp.sum(cp.log(max_charge - c))
                ))

            # TODO: adjust code to also include the possibility of a joint charge/discharge limit in log-barrier

        if cyclic_bc:
            for i in range(bs):
                constraints.append(A_last @ s[i] == soc_0 * a_last)

        prob = cp.Problem(objective=objective, constraints=constraints)

        self.prob = prob
        self.params = [soc_0, price]
        self.vars = [net_discharge, d, c, s]
        self.constraints = constraints

    def get_opti_matrices(self):
        """
        Function returning a list of matrices that can be used to define storage system-related optimization problems in matrix form
        """

        D_out = self.OP_params['lookahead']
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

    def get_opti_problem(self):

        return self.prob, self.params, self.vars

    def get_var_values(self):

        var_values = []
        for v in self.vars:
            var_values.append(v.value)

        #Add mu
        mu_end = self.constraints[7].dual_value
        mu_start = self.constraints[6].dual_value
        try:
            mu_end[0] = mu_start[0]
        except:
            print(mu_end)
            print(mu_start)
        var_values.append(mu_end)
        return var_values

    def set_params_opti(self,param_values):

        for (i,p_val) in enumerate(param_values):
            self.params[i].value = p_val