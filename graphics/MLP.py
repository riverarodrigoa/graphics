from __future__ import print_function, division
import numpy as np
from netCDF4 import Dataset
import os
import sys
from multiprocessing import Process, Queue
from scipy.optimize import fmin_l_bfgs_b as bfgs
import pandas as pd


class MLP:
    def __init__(self, hidden_units):
        self.input_units = 1
        self.hidden_units = hidden_units
        self.output_units = 1
        self.n_par = 1      # Number of parameters
        self.n_el_sig = 1   # Number of parameters (conexion input layer/hidden layer)
        self.n_el_lin = 1   # Number of parmeters (conexion hidden layer/output layer)
        self.weights = []
        self.max_weight_ww = [-4, 4]
        self.data_file = ''
        self.result_dir = ''
        self.name_var_in = []
        self.n_var_in = 1
        self.name_var_out = []
        self.training_index = []
        self.validation_index = []
        self.n_all = 1
        self.n_loops = 1000
        self.n_procs = 4
        self.n_opt = 0
        self.regul_fctr = 0.1
        self.n_samples = 50
        self.sample = 1
        self.validation_rate = 0.3
        self.min_vars = {}
        self.max_vars = {}
        self.all_data = []
        self.training_data = []
        self.validation_data = []
        self.norm_training_data = []
        self.norm_training_inputs = []
        self.norm_validation_data = []
        self.norm_validation_inputs = []
        self.norm_all_inputs = []
        self.opt_xx = []
        self.output_evaluation = []
        self.real_output = []
        self.training_rmse = []
        self.validation_rmse = []
        self.n_validation = 0
        self.n_training = 0
        self.opt_outputs_ann =[]
        self.opt_training_rmse = 0
        self.opt_validation_rmse = 0

    def __forward(self, weights, inputs):
        # ntime = inputs.shape[0]
        n_el_sig = (self.input_units + 1) * self.hidden_units
        # n_el_lin = self.hidden_units + 1
        sig_synaptic_weights = np.reshape(weights[:n_el_sig], (self.input_units + 1, self.hidden_units))
        layer2_synaptic_weights = np.reshape(weights[n_el_sig:], (self.hidden_units + 1, 1))
        # Summer for the sigmoid transfer function
        sig_inputs = np.dot(inputs, sig_synaptic_weights[:-1, :])
        # Bias or offset
        sig_inputs = sig_inputs - sig_synaptic_weights[-1, :]
        # Sigmoid outputs
        sig_output_from_layer1 = (np.exp(sig_inputs) - np.exp(-sig_inputs)) / (np.exp(sig_inputs) + np.exp(-sig_inputs))

        # LINEAR SUMMER
        lin_inputs = np.dot(sig_output_from_layer1, layer2_synaptic_weights[:-1, :])
        lin_inputs = lin_inputs - layer2_synaptic_weights[-1, :]

        return lin_inputs

    def __costf_fct(self, weights, inputs, outputs):
        tmp_fct_obs = ((outputs - self.__forward(weights, inputs)) ** 2).mean()

        tmp_fct_par = (weights ** 2).mean()
        costf = tmp_fct_obs + self.regul_fctr * tmp_fct_par
        return costf

    def bfgs_fct(self, qq, iproc):
        try:
            # Definition of a callback function to store the values of the cost function
            tmp_training_costf_arr = []
            tmp_validation_costf_arr = []
            xx_arr = []

            def callback_fct(xxk):
                # We remove the parameter contribution to the cost function
                tmp_training_costf_arr.append(((self.norm_training_data - self.__forward(xxk, self.norm_training_inputs)) ** 2).mean())
                tmp_validation_costf_arr.append(((self.norm_validation_data - self.__forward(xxk, self.norm_validation_inputs)) ** 2).mean())
                xx_arr.append(xxk)

            # --- Initialize BFGS algorithm parameters                   --
            np.random.seed(iproc)
            xx0 = np.random.uniform(-0.5, 0.5, self.n_par)
            bnds_list = []
            for iel in range(self.n_par):
                bnds_list.append((self.max_weight_ww[0], self.max_weight_ww[1]))
            # Calling BFGS
            # xx_opt, costf, diag_dict =
            bfgs(self.__costf_fct, xx0, fprime=None, args=(self.norm_training_inputs, self.norm_training_data),
                 approx_grad=True, bounds=bnds_list, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-06,
                 iprint=0, maxfun=50000000, maxiter=self.n_loops, disp=None, callback=callback_fct)

            # We send to the main the optimal parameter set
            xx_arr = np.array(xx_arr)
            qq.put([xx_arr, tmp_training_costf_arr, tmp_validation_costf_arr])

        except():
            print('ERROR ON SUBPROCESS: ', sys.exc_info()[1])
            qq.put(0)

    def update_n_params(self):
        self.n_par = self.hidden_units * (self.n_var_in + 1) + self.hidden_units + 1  # 10*(5+1)+(10+1)=71
        self.n_el_sig = self.hidden_units * (self.n_var_in + 1)                       # 10*(5+1) = 60
        self.n_el_lin = self.hidden_units + 1                                         # 10+1 = 11
        # print('PARAMS UPDATED: {} | {} | {}'.format(self.n_par, self.n_el_sig, self.n_el_lin))

    def __get_validation_index__(self, n, validation_rate, offset):
        n_validation = int(validation_rate*n)
        i0_min = offset
        i0_max = n - offset - n_validation
        i0 = int(i0_min + (self.sample - 1) * ((i0_max - i0_min) / self.n_samples))
        i1 = int(i0 + n_validation)
        validation_index = range(i0, i1)
        return validation_index

    def __load_data__(self, data_file, input_variables, output_variables, sample_test=1, nsamples=50, normalisation=2, train_val_test_ratio=(0.7, 0.1, 0.2), offset_data=60 * 24 * 3, encoding='latin1'):
        self.name_var_in = input_variables
        self.name_var_out = output_variables
        self.input_units = len(self.name_var_in)
        self.output_units = len(output_variables)
        print(self.input_units, self.output_units)
        self.update_n_params()
        self.sample = sample_test
        self.n_samples = nsamples
        variables = np.load(data_file, encoding=encoding, allow_pickle=True)  # nn_vars_all
        data = pd.DataFrame(variables.item())
        data['Time'] = pd.to_datetime(data['Time'], format="%Y-%m-%d_%H:%M:%S")
        data.set_index('Time', inplace=True)
        self.dataset = data
        self.data_n = self.__normalisation__(data=self.dataset, type_norm=normalisation)
        self.train_set, self.val_set, self.test_set = self.__split_dataset__(data=self.data_n, train_val_test_ratio=train_val_test_ratio, offset_data=offset_data)
        self.norm_training_inputs = self.train_set.loc[:, self.name_var_in].to_numpy()
        self.norm_training_data = self.train_set.loc[:, self.name_var_out].to_numpy()
        self.norm_validation_inputs = self.val_set.loc[:, self.name_var_in].to_numpy()
        self.norm_validation_data = self.val_set.loc[:, self.name_var_out].to_numpy()
        self.norm_test_inputs = self.test_set.loc[:, self.name_var_in].to_numpy()
        self.norm_test_data = self.test_set.loc[:, self.name_var_out].to_numpy()

        print(self.norm_training_inputs)

        # Get min & max values for all the columns
        set_vars = np.load(data_file, allow_pickle=True)
        set_vars = set_vars.flat[0]
        min_vars = {}
        max_vars = {}
        for name_var in set_vars:
            if name_var != 'Time':
                min_vars[name_var] = np.min(set_vars[name_var])
                max_vars[name_var] = np.max(set_vars[name_var])

        self.min_vars = min_vars
        self.max_vars = max_vars

        if self.n_opt == 0:  # What does this?????
            self.n_opt = (self.n_procs - 1)

    @staticmethod
    def __normalisation__(data, type_norm=2):
        assert isinstance(data, pd.DataFrame), "[ERROR]: X must be a pandas DataFrame."
        stats = data.describe().transpose()

        if type_norm == 1:
            return (data - stats['mean']) / stats['std']
        elif type_norm == 2:
            return 2 * (data - stats['min']) / (stats['max'] - stats['min']) - 1
        else:
            return data

    @staticmethod
    def __get_period__(x, ix, rate=0.3, offset=60 * 24 * 3, samples=50):
        n = len(ix)  # 100
        n_val = int(len(ix) * rate)  # 100*0.3= 30 len test set
        imin = offset  # 0
        imax = n - offset - n_val  # 100-0-30=70
        n_sample = (imax - imin) / samples  # (70-0)/2=35

        start = int(imin + x * n_sample)  # 0+0*35= 0
        end = int(start + n_val)  # 0+35=35

        return ix[start], ix[end]

    def __split_dataset__(self, data, train_val_test_ratio, offset_data):
        start, end = self.__get_period__(x=self.sample, ix=data.index, rate=train_val_test_ratio[2], offset=offset_data, samples=self.n_samples)  # Test set
        test_set = data.loc[start:end, :]
        train_val_set = data.loc[(data.index < start) | (data.index > end), :]
        n = len(train_val_set.index)
        n_val = int(n*train_val_test_ratio[1])
        train_set = train_val_set.iloc[:n-n_val, :]
        val_set = train_val_set.iloc[-n_val:, :]
        return train_set, val_set, test_set

    def evaluate(self):
        norm_output = self.__forward(self.opt_xx, self.norm_test_inputs)
        self.output_evaluation = np.squeeze((self.max_vars[self.name_var_out] - self.min_vars[self.name_var_out]) * (norm_output + 1.) / 2. + self.min_vars[self.name_var_out])
        self.real_output = self.dataset.loc[self.test_set.index, self.name_var_out].values
        rmse_evaluation = np.sqrt(((self.real_output - self.output_evaluation) ** 2).mean())
        print("RMSE[Test set]: {}".format(rmse_evaluation))
        return rmse_evaluation

    def load_data(self, data_file, name_var_in, validation_rate, sample=1):
        self.data_file = data_file
        self.name_var_in = name_var_in    # Name of the input variables
        self.n_var_in = len(self.name_var_in)
        self.name_var_out = 'CH4d_ppm'
        self.input_units = len(self.name_var_in)
        self.output_units = len([self.name_var_out])
        self.validation_rate = validation_rate
        set_vars = np.load(data_file, encoding='latin1', allow_pickle=True)
        set_vars = set_vars.flat[0]
        self.n_all = len(set_vars['Time'])
        self.sample = sample
        self.validation_index = self.__get_validation_index__(n=self.n_all, validation_rate=self.validation_rate, offset=10)
        self.update_n_params()
        # n_neurons_sig = n_var_in * 4  # self.hidden_units
        if self.n_opt == 0:                     # What does this?????
            self.n_opt = (self.n_procs - 1)

        training_index = np.delete(np.arange(self.n_all), self.validation_index)

        validation_index = np.array(self.validation_index, dtype=np.int32)
        training_index = np.array(training_index, dtype=np.int32)

        self.training_index = training_index
        self.validation_index = validation_index

        self.n_validation = len(validation_index)
        self.n_training = len(training_index)

        # NORM VARIABLES ----------------------------------
        min_vars = {}
        max_vars = {}
        for name_var in set_vars:
            if name_var != 'Time':
                min_vars[name_var] = np.min(set_vars[name_var])
                max_vars[name_var] = np.max(set_vars[name_var])

        self.min_vars = min_vars
        self.max_vars = max_vars

        # The training set. We have n_training examples, each consisting of n_var_in input values
        # and 1 output value.
        all_inputs = np.zeros((self.n_all, self.n_var_in))
        all_data = np.zeros(self.n_all)
        training_inputs = np.zeros((self.n_training, self.n_var_in))
        training_data = np.zeros(self.n_training)
        validation_inputs = np.zeros((self.n_validation, self.n_var_in))
        validation_data = np.zeros(self.n_validation)

        self.all_data = all_data
        self.training_data = training_data
        self.validation_data = validation_data

        ivar = 0
        for name_var in name_var_in:
            training_inputs[:, ivar] = set_vars[name_var][training_index]
            validation_inputs[:, ivar] = set_vars[name_var][validation_index]
            all_inputs[:, ivar] = set_vars[name_var][:]
            ivar += 1
        # for name_var in name_var_in:
        self.training_data = set_vars[self.name_var_out][training_index]
        self.validation_data = set_vars[self.name_var_out][validation_index]
        self.all_data = set_vars[self.name_var_out][:]

        # We normalize all the variables between -1 and 1
        self.norm_training_inputs = training_inputs.copy()
        self.norm_training_data = self.training_data.copy()
        self.norm_validation_inputs = validation_inputs.copy()
        self.norm_validation_data = self.validation_data.copy()
        self.norm_all_inputs = all_inputs.copy()
        # norm_all_data = all_data.copy()

        for ivar in range(self.n_var_in):
            self.norm_training_inputs[:, ivar] = 2. * (training_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1
            self.norm_validation_inputs[:, ivar] = 2. * (validation_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1
            self.norm_all_inputs[:, ivar] = 2. * (all_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1

        self.norm_training_data = 2 * (self.training_data - min_vars[self.name_var_out]) / (max_vars[self.name_var_out] - min_vars[self.name_var_out]) - 1  # .
        self.norm_training_data = np.reshape(self.norm_training_data, [self.n_training, 1])
        self.norm_validation_data = 2 * (self.validation_data - min_vars[self.name_var_out]) / (max_vars[self.name_var_out] - min_vars[self.name_var_out]) - 1  # .
        self.norm_validation_data = np.reshape(self.norm_validation_data, [self.n_validation, 1])
        # norm_all_data = 2 * (all_data - min_vars[self.name_var_out]) / (max_vars[self.name_var_out] - min_vars[self.name_var_out]) - 1.
        # norm_all_data = np.reshape(norm_all_data, [n_all, 1])

    def load_data2(self, df, name_var_in, validation_rate, sample=1):  # data_file, name_var_in, validation_rate, sample=1):
        # self.data_file = df
        self.name_var_in = name_var_in    # Name of the input variables
        self.n_var_in = len(self.name_var_in)
        self.name_var_out = 'CH4d_ppm'
        self.input_units = len(self.name_var_in)
        self.output_units = len([self.name_var_out])
        self.validation_rate = validation_rate
        # set_vars = np.load(data_file, encoding='latin1', allow_pickle=True)
        # set_vars = set_vars.flat[0]
        set_vars = df
        self.n_all = len(set_vars['Time'])
        self.sample = sample
        self.validation_index = self.__get_validation_index__(n=self.n_all, validation_rate=self.validation_rate, offset=10)
        self.update_n_params()
        # n_neurons_sig = n_var_in * 4  # self.hidden_units
        if self.n_opt == 0:                     # What does this?????
            self.n_opt = (self.n_procs - 1)

        training_index = np.delete(np.arange(self.n_all), self.validation_index)

        validation_index = np.array(self.validation_index, dtype=np.int32)
        training_index = np.array(training_index, dtype=np.int32)

        self.training_index = training_index
        self.validation_index = validation_index

        self.n_validation = len(validation_index)
        self.n_training = len(training_index)

        # NORM VARIABLES ----------------------------------
        min_vars = {}
        max_vars = {}
        for name_var in set_vars:
            if name_var != 'Time':
                min_vars[name_var] = np.min(set_vars[name_var])
                max_vars[name_var] = np.max(set_vars[name_var])

        self.min_vars = min_vars
        self.max_vars = max_vars

        # The training set. We have n_training examples, each consisting of n_var_in input values
        # and 1 output value.
        all_inputs = np.zeros((self.n_all, self.n_var_in))
        all_data = np.zeros(self.n_all)
        training_inputs = np.zeros((self.n_training, self.n_var_in))
        training_data = np.zeros(self.n_training)
        validation_inputs = np.zeros((self.n_validation, self.n_var_in))
        validation_data = np.zeros(self.n_validation)

        self.all_data = all_data
        self.training_data = training_data
        self.validation_data = validation_data

        ivar = 0
        for name_var in name_var_in:
            training_inputs[:, ivar] = set_vars[name_var][training_index]
            validation_inputs[:, ivar] = set_vars[name_var][validation_index]
            all_inputs[:, ivar] = set_vars[name_var][:]
            ivar += 1
        # for name_var in name_var_in:
        self.training_data = set_vars[self.name_var_out][training_index]
        self.validation_data = set_vars[self.name_var_out][validation_index]
        self.all_data = set_vars[self.name_var_out][:]

        # We normalize all the variables between -1 and 1
        self.norm_training_inputs = training_inputs.copy()
        self.norm_training_data = self.training_data.copy()
        self.norm_validation_inputs = validation_inputs.copy()
        self.norm_validation_data = self.validation_data.copy()
        self.norm_all_inputs = all_inputs.copy()
        # norm_all_data = all_data.copy()

        for ivar in range(self.n_var_in):
            self.norm_training_inputs[:, ivar] = 2. * (training_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1
            self.norm_validation_inputs[:, ivar] = 2. * (validation_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1
            self.norm_all_inputs[:, ivar] = 2. * (all_inputs[:, ivar] - min_vars[name_var_in[ivar]]) / (max_vars[name_var_in[ivar]] - min_vars[name_var_in[ivar]]) - 1

        self.norm_training_data = 2 * (self.training_data - min_vars[self.name_var_out]) / (max_vars[self.name_var_out] - min_vars[self.name_var_out]) - 1  # .
        self.norm_training_data = np.reshape(self.norm_training_data, [self.n_training, 1])
        self.norm_validation_data = 2 * (self.validation_data - min_vars[self.name_var_out]) / (max_vars[self.name_var_out] - min_vars[self.name_var_out]) - 1  # .
        self.norm_validation_data = np.reshape(self.norm_validation_data, [self.n_validation, 1])

    def train(self, regul_fctr, n_loops, n_procs):
        self.regul_fctr = regul_fctr
        self.n_loops = n_loops
        self.n_procs = n_procs

        training_costf_arr = np.zeros((self.n_opt, self.n_loops))  # <-the +1 stands for the mean
        validation_costf_arr = np.zeros((self.n_opt, self.n_loops))  # <-the +1 stands for the mean
        xx_arr = np.zeros((self.n_opt, self.n_loops, self.n_par))
        xx_opt_arr = np.zeros((self.n_opt, self.n_par))

        iopt = 0
        while iopt < self.n_opt:
            qq_arr = []
            process_pool = []
            # Sending data to the processors
            for iproc in range(n_procs - 1):
                qq = Queue()
                qq_arr.append(qq)
                pp = Process(target=self.bfgs_fct, args=(qq, iproc))
                pp.start()
                process_pool.append(pp)

            for iproc in range(n_procs - 1):
                if iopt == self.n_opt:                   # What does this?
                    continue
                proc_data = qq_arr[iproc].get()
                tmp_n_loops = len(proc_data[1])
                xx_arr[iopt, :tmp_n_loops, :] = proc_data[0]
                xx_opt_arr[iopt, :] = xx_arr[iproc, tmp_n_loops - 1, :]
                training_costf_arr[iopt, :tmp_n_loops] = proc_data[1]
                validation_costf_arr[iopt, :tmp_n_loops] = proc_data[2]
                iopt += 1

            # We terminate the processes
            for pp in process_pool:
                pp.join()

        #  We store the outputs of the ANN weights at the end of each optimization
        #  And we compute the associated RMSE's
        outputs_ann = np.zeros((self.n_opt + 1, self.n_all))  # <-the +1 stands for the mean
        training_rmse = np.zeros(self.n_opt + 1)
        validation_rmse = np.zeros(self.n_opt + 1)
        mean_outputs_ann = 0
        for iopt in range(self.n_opt):
            xx_opt = xx_opt_arr[iopt, :]
            # Computing the optimized nn outputs
            tmp_outputs_ann = self.__forward(xx_opt, self.norm_all_inputs)                                              # check ( maybe here could go the test set)
            # Computing the "real" outputs of the NN (unnormalized outputs)
            tmp_outputs_ann = (self.max_vars[self.name_var_out] - self.min_vars[self.name_var_out]) * (tmp_outputs_ann + 1.) / 2. + self.min_vars[self.name_var_out]
            tmp_outputs_ann = np.squeeze(tmp_outputs_ann)
            outputs_ann[iopt, :] = tmp_outputs_ann

            mean_outputs_ann += tmp_outputs_ann

            training_rmse[iopt] = np.sqrt(((self.training_data - tmp_outputs_ann[self.training_index]) ** 2).mean())
            validation_rmse[iopt] = np.sqrt(((self.validation_data - tmp_outputs_ann[self.validation_index]) ** 2).mean())

        # The optimal ANN is the one which minimizes the validation costf
        ii_opt = np.argmin(validation_rmse[:-1])
        self.opt_outputs_ann = outputs_ann[ii_opt]
        opt_xx = xx_opt_arr[ii_opt, :]
        self.opt_training_rmse = training_rmse[ii_opt]
        self.opt_validation_rmse = validation_rmse[ii_opt]
        self.opt_xx = opt_xx

        # mean_outputs_ann /= self.n_opt

        # self.training_rmse[self.n_opt] = np.sqrt(((self.training_data - mean_outputs_ann[self.training_index]) ** 2).mean())
        # self.validation_rmse[self.n_opt] = np.sqrt(((self.validation_data - mean_outputs_ann[self.validation_index]) ** 2).mean())
        # # mean_training_rmse = training_rmse[self.n_opt]
        # mean_validation_rmse = validation_rmse[self.n_opt]

        print("[SAMPLE: {}] Training RMSE: {} | Vallidation RMSE: {}".format(self.sample, self.opt_training_rmse, self.opt_validation_rmse))

    def get_model_params(self):
        print("MODEL CONFIGURATION", '\n',
              "-------------------", '\n',
              "Input Units: {}".format(self.input_units), '\n',
              "Hidden Units: {}".format(self.hidden_units), '\n',
              "Output Units: {}".format(self.output_units))

        return self.input_units, self.hidden_units, self.output_units

    def save_results(self, result_path, name_run):
        self.result_dir = result_path
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        result_file = self.result_dir + name_run + '.nc'
        # len_data_file = len(self.data_file)
        # Strings with the input var names
        tmp_str = ''
        for name_var in self.name_var_in:
            tmp_str = tmp_str + '+' + name_var
        name_var_str = tmp_str[1:]
        len_name_var_str = len(name_var_str)

        wfile = Dataset(result_file, 'w')
        wfile.createDimension('n_var_in', self.n_var_in)
        wfile.createDimension('k1', 1)
        wfile.createDimension('n_neurons_sig', self.hidden_units)
        wfile.createDimension('n_training', self.n_training)
        wfile.createDimension('n_validation', self.n_validation)
        wfile.createDimension('n_loops', self.n_loops)
        wfile.createDimension('n_par', self.n_par)
        wfile.createDimension('n_opt', self.n_opt)
        wfile.createDimension('n_opt_mean', self.n_opt + 1)  # The +1 stands for the mean model
        wfile.createDimension('n_all', self.n_all)
        wfile.createDimension('len_name_var_str', len_name_var_str)
        # wfile.createDimension('len_data_file', len_data_file)
        wfile.comments = 'Variables Normalization: min-max'
        ncvar = wfile.createVariable('xx_opt', 'f', ('n_par',))
        ncvar[:] = self.opt_xx
        ncvar.comments = 'ANN WEIGHTS VALUES at convergence for the optimal optimization'
        ncvar = wfile.createVariable('regul_fctr', 'f', ('k1',))
        ncvar[:] = self.regul_fctr
        ncvar = wfile.createVariable('opt_training_rmse', 'f', ('k1',))
        ncvar[:] = self.opt_training_rmse
        ncvar = wfile.createVariable('opt_validation_rmse', 'f', ('k1',))
        ncvar[:] = self.opt_validation_rmse
        ncvar = wfile.createVariable('opt_outputs_ann', 'f', ('n_all',))
        ncvar[:] = self.opt_outputs_ann
        ncvar.comments = 'ANN outputs over the whole data period for the optimal model'
        ncvar = wfile.createVariable('all_data', 'f', ('n_all',))
        ncvar[:] = self.all_data
        ncvar.comments = 'data values over the whole data period'
        ncvar = wfile.createVariable('name_var_str', 'c', ('len_name_var_str',))
        ncvar[:] = name_var_str
        ncvar = wfile.createVariable('data_file', 'c', ('len_data_file',))
        ncvar[:] = self.data_file
        ncvar = wfile.createVariable('training_index', 'i', ('n_training',))
        ncvar[:] = self.training_index
        ncvar = wfile.createVariable('validation_index', 'i', ('n_validation',))
        ncvar[:] = self.validation_index
        wfile.close()


def main(var1, var2):

    path = 'C:/Users/rrivera/Documents/PROJECTS/nn_diego/'
    if not os.path.exists(path):
        path = '/Users/rodrigo/Documents/Projects/nn_diego/'
    if not os.path.exists(path):
        path = '/home/users/rrivera/NNDiego/'
    datafile = path + 'data/nn_vars_filter.npy'
    var_comb = var1
    name_case = var2
    print(var_comb, name_case)
    model = MLP(10)
    for k in range(len(name_case)):
        for i in range(1, 51):
            model.load_data(data_file=datafile, name_var_in=var_comb[k],
                            validation_rate=0.3, sample=i)
            model.get_model_params()
            model.train(regul_fctr=0.1, n_loops=1000, n_procs=8)
            model.save_results(result_path=path + 'results/sens_test/'+name_case[k], name_run=name_case[k] + '_' + str(i))


if __name__ == '__main__':
    print('test')
