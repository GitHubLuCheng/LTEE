import random

from math import sqrt
import cfr.cfr_net_x as cfr
from cfr.util import *
from sklearn.metrics import mean_squared_error
"""
Code is adapted from cfrnet implementing the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976
"""

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-8, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 1e-6, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate_train', 0.001, """Training learning rate. """)
tf.app.flags.DEFINE_float('lrate_test', 0.0001, """Testing learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('batch_size_test', 100, """Batch size for fine tune. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 1, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none',
                           """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 1000, """Number of iterations training. """)
tf.app.flags.DEFINE_integer('iterations_tune', 500, """Number of iterations testing. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 10, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'wass',
                           """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv', 0, """Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 10, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1,
                            """Number of iterations between pre

                            diction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1,
                            """Whether to reweight sample for prediction loss with average treatment probability. """)

NUM_ITERATIONS_PER_DECAY = 100
t0=10
x_DIM=25
result_file = open('data/IHDP/' + str(t0) + '.txt', 'w')

for j in range(1, 11):
    TY = np.loadtxt('data/IHDP/csv/ihdp_npci_' + str(j) + '.csv', delimiter=',')
    matrix = TY[:, 5:]
    N = TY.shape[0]

    out_treat = np.loadtxt('data/IHDP/Series_y_' + str(j) + '.txt', delimiter=',')
    ts = out_treat[:, 0]
    ts = np.reshape(ts, (N, 1))
    ys = np.concatenate((out_treat[:, 1:(t0 + 1)], out_treat[:, -1].reshape(N, 1)), axis=1)

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, t0, x_DIM], name='x')  # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatment
    y_ = tf.placeholder("float", shape=[None, t0 + 1], name='y_')  # Outcomes

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    test = tf.placeholder("float", name='test')
    lr_input = tf.placeholder('float')

    dims = [x_DIM, FLAGS.dim_in, FLAGS.dim_out]
    CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims, test,t0)

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(lr_input, global_step, \
                                    NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    train_step = opt.minimize(CFR.tot_loss, global_step=global_step)

    '''train validation test split'''
    from sklearn.model_selection import train_test_split

    matrix_rep = np.repeat(matrix[:, np.newaxis, :], t0, axis=1)
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(matrix_rep, ys, ts, test_size=0.2)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p_treated_train = np.mean(t_train)
    p_treated_test = np.mean(t_test)
    dict_factual = {CFR.x: X_train, CFR.t: t_train, CFR.y_: y_train, \
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
                    CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated_train, CFR.test: 1, lr_input: FLAGS.lrate_train}

    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], \
                                          feed_dict=dict_factual)

    losses.append([obj_loss, f_error, imb_err])
    objnan = False

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = X_train[I, :]
        t_batch = t_train[I]
        y_batch = y_train[I, :]

        sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                                        CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                        CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda,
                                        CFR.p_t: p_treated_train, CFR.test: 1, lr_input: FLAGS.lrate_train})

        if i % FLAGS.output_delay == 0 or i == FLAGS.iterations - 1:
            obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                                                  feed_dict=dict_factual)

            losses.append([obj_loss, f_error, imb_err])
            loss_str = str(i) + '\tObj: %.3f,\tPred: %.3f,\tImb: %.2g' \
                       % (obj_loss, f_error, imb_err)

            if FLAGS.loss == 'log':
                y_pred = sess.run(CFR.output, feed_dict={CFR.x: x_batch, \
                                                         CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred = 1.0 * (y_pred > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc
                # print loss_str

            if FLAGS.loss == 'l2':
                y_pred = sess.run(CFR.output, feed_dict={CFR.x: x_batch, \
                                                         CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                rmse = sqrt(mean_squared_error(y_batch, y_pred))
                loss_str += ',\tRMSE: %.2f' % rmse
                print loss_str

            '''if np.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True'''

    y_pred = sess.run(CFR.output, feed_dict={CFR.x: X_test, \
                                             CFR.t: t_test, CFR.do_in: 1.0, CFR.do_out: 1.0})
    y_pred_CF = sess.run(CFR.output, feed_dict={CFR.x: X_test, \
                                                CFR.t: 1 - t_test, CFR.do_in: 1.0, CFR.do_out: 1.0})
    y_pred = y_pred[:, t0]
    y_pred_CF = y_pred_CF[:, t0]

    groundtruth = np.loadtxt('data/IHDP/Series_groundtruth_' + str(j) + '.txt')
    groundtruth_T = groundtruth[-1]
    if FLAGS.loss == 'l2':
        effect = np.mean(abs(y_pred - y_pred_CF), axis=0)
        effe_str = 'Abs error: %.2f' % abs(effect - groundtruth_T)
        print effe_str
        result_file.write(effe_str + '\n')
    tf.reset_default_graph()
