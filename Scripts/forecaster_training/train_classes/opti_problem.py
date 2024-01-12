import numpy as np
import cvxpy as cp

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

    def get_opti_problem(self):
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

        return prob, [price], [net_discharge, d, c, s]

