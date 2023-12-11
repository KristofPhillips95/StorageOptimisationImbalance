import numpy as np
import cvxpy as cp

class OptiProblem():
    """
    Class of Optimization problems
    """
    def __init__(self,OP_params):

        super(OptiProblem,self).__init__()
        self.OP_params = OP_params
        self.init_opti_problem()

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

        self.prob.solve(solver=cp.ECOS)

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

        self.prob = prob
        self.params = [soc_0,price]
        self.vars = [net_discharge,d,c,s]

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
        return var_values

    def set_params_opti(self,param_values):

        for (i,p_val) in enumerate(param_values):
            self.params[i].value = p_val