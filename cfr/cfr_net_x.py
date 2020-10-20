import tensorflow as tf
import numpy as np

from util import *
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
class cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, test, t0):
        self.variables = {}
        self.wd_loss = 0
        self.attention_size=50
        self.seq_len_ph=t0
        self.test=test

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]


        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
        ''' Construct input/representation layers '''
        self.phi=tf.layers.dense(x,dim_in,activation=tf.nn.relu,kernel_regularizer=regularizer)
        ''' Regularization '''
        self.wd_loss+=tf.losses.get_regularization_loss()

        rnn_outputs, _ = bi_rnn(GRUCell(dim_in), GRUCell(dim_in),
                                inputs=self.phi, dtype=tf.float32)

        h_rep = tf.concat([rnn_outputs[0],rnn_outputs[1]],2) #`[batch_size, max_time, cell.output_size]`

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0*h_rep

        ''' Construct ouput layers '''
        y = self._build_output_graph(rnn_outputs, t, dim_in, dim_out, do_out, FLAGS) # y contains all the predictions in the sequence

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        y_seq=y[:,0:self.seq_len_ph]
        y_T=y[:,self.seq_len_ph]
        y_seq_ = y_[:, 0:self.seq_len_ph]
        y_T_ = y_[:, self.seq_len_ph]
        if FLAGS.loss == 'l1':
            risk1 = tf.reduce_mean(sample_weight*tf.abs(y_seq_-y_seq))
            risk2 = tf.reduce_mean(sample_weight * tf.abs(y_T_ - y_T))
            risk = risk1 + self.test*risk2
            pred_error_1 = tf.reduce_mean(tf.abs(y_seq_ - y_seq))
            pred_error_2 = tf.reduce_mean(tf.abs(y_T_ - y_T))
            pred_error = pred_error_1 + self.test * pred_error_2
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            res_seq = res[:, 0:self.seq_len_ph]
            res_T = res[:, self.seq_len_ph]
            risk1 = -tf.reduce_mean(sample_weight*res_seq)
            risk2 = -tf.reduce_mean(sample_weight * res_T)
            risk=risk1+self.test *risk2

            pred_error1 = -tf.reduce_mean(res_seq)
            pred_error2 = -tf.reduce_mean(res_T)
            pred_error= pred_error1+self.test *pred_error2
        else:
            risk1 = tf.reduce_mean(sample_weight*tf.square(y_seq_ - y_seq))
            risk2 = tf.reduce_mean(sample_weight * tf.square(y_T_ - y_T))
            risk = risk1 + self.test * risk2
            pred_error1 = tf.sqrt(tf.reduce_mean(tf.square(y_seq_ - y_seq)))
            pred_error2 = tf.sqrt(tf.reduce_mean(tf.square(y_T_ - y_T)))
            pred_error = pred_error1 + self.test * pred_error2

        ''' Imbalance error '''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma)
            imb_error = r_alpha*imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist = wasserstein(self.seq_len_ph, h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            #self.imb_mat = imb_mat # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
            imb_error = r_alpha * imb_dist
            #self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
            imb_error = r_alpha * imb_dist

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha>0:
            tot_error = tot_error + imb_error

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss


        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN outputs with Attention vector.

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega=self._create_variable(tf.random_normal([hidden_size, attention_size],
                                                              stddev=FLAGS.weight_init / np.sqrt(hidden_size)), 'w_omega')
        b_omega=self._create_variable(tf.random_normal([attention_size],
                                                              stddev=FLAGS.weight_init / np.sqrt(attention_size)), 'b_omega')
        u_omega = self._create_variable(tf.random_normal([attention_size],
                                                              stddev=FLAGS.weight_init / np.sqrt(attention_size)), 'u_omega')

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def _build_output(self, phi, rep_fw,rep_bw, dim_in, dim_out, do_out, FLAGS):

        h_input = tf.concat([rep_fw,rep_bw],2) #`[batch_size, max_time, cell.output_size]`
        if FLAGS.normalization == 'divide':
            h_input = h_input / safe_sqrt(tf.reduce_sum(tf.square(h_input), axis=1, keep_dims=True))
        else:
            h_input = 1.0*h_input

        y_seq = tf.layers.dense(h_input,1,activation=tf.nn.relu)

        attention_output = self.attention(h_input, self.attention_size)
        h_out = [attention_output]
        dims = [2 * dim_in] + ([dim_out] * FLAGS.n_out)

        weights_out = []
        biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=FLAGS.weight_init / np.sqrt(dims[i])),
                'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out+dim_in, 1],
                                                              stddev=FLAGS.weight_init / np.sqrt(dim_out)),
                                             'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(
                tf.slice(weights_pred, [0, 0], [dim_out - 1, 1]))  # don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        h_pred = h_out[-1]
        h_pred_full = tf.concat([phi,h_pred],axis=1)#concatenate representation of x and surrogate representation
        y = tf.matmul(h_pred_full, weights_pred) + bias_pred
        y_seq = tf.squeeze(y_seq, [2])
        y_ls = tf.concat([y_seq, y], 1)
        return y_ls

    def _build_output_graph(self, rnn_outputs, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        i0 = tf.to_int32(tf.where(t < 1)[:,0])
        i1 = tf.to_int32(tf.where(t > 0)[:,0])

        rep_fw=rnn_outputs[0]
        rep_bw=rnn_outputs[1]

        rep_fw0 = tf.gather(rep_fw, i0)
        rep_fw1 = tf.gather(rep_fw, i1)
        rep_bw0 = tf.gather(rep_bw, i0)
        rep_bw1 = tf.gather(rep_bw, i1)
        phi_0 = tf.gather(self.phi[:,0,:], i0)
        phi_1 = tf.gather(self.phi[:,0,:], i1)

        y0 = self._build_output(phi_0, rep_fw0,rep_bw0, dim_in, dim_out, do_out, FLAGS)
        y1 = self._build_output(phi_1, rep_fw1,rep_bw1, dim_in, dim_out, do_out, FLAGS)

        y = tf.dynamic_stitch([i0, i1], [y0, y1])
        return y
