# Import libraries
# import tensorflow as tf # tensorflow_version 1.x is needed

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import copy
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy.io import loadmat



filename1 = '6tx6rx4ants50disBS20-80dis_user0indoor_H_realizations_training.mat'
data = loadmat(filename1)
H = data['H_realizations'] * 10**4

# 读取信道相关信息
nr_of_users = data['no_rx'].squeeze()#6
nr_of_antennas = data['M'].squeeze()#4  #nr_of_antennas = data['MN'].squeeze() #UPA
nr_of_BSs = data['no_tx'].squeeze()#4
nr_of_BS_antennas = nr_of_antennas * nr_of_BSs
filename2 = './test_data/H_realizations_test_3.mat'
data = loadmat(filename2)
H_test = data['H_realizations'] * 10**4
scheduled_users = []
for ii in range(nr_of_users):
    scheduled_users.append(ii)  # array of scheduled users. Note that we schedule all the users.
epsilon = 0.0001 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001 # used to end the bisection search in the WMMSE algorithm in Shi et al.
total_power = 1 # power constraint in the weighted sum rate maximization problem
noise_power = 1
path_loss_option = True # used to add a random path loss (drawn from a uniform distribution) to the channel of each user
path_loss_range = [-5,5] # interval of the uniform distribution from which the path loss if drawn (in dB)
nr_of_samples_per_batch = 24 # mini-batch的大小
nr_of_batches_training = np.int(np.size(H,0) / nr_of_samples_per_batch) # 训练的总mini-batch数
nr_of_batches_test = np.int(np.size(H_test,0) / nr_of_samples_per_batch) # 测试的总mini-batch数

nr_of_iterations = 2 # WMMSE algorithm 的迭代次数
nr_of_iterations_nn = 2 #  深度展开神经层的层数

# User weights in the weighted sum rate
user_weights = np.reshape(np.ones(nr_of_users*nr_of_samples_per_batch),(nr_of_samples_per_batch,nr_of_users,1))
user_weights_for_regular_WMMSE = np.ones(nr_of_users)


def compute_norm_of_complex_array(x):
    result = np.sqrt(np.sum((np.absolute(x)) ** 2))  #np.sqrt平方根函数 np.absolute 绝对值函数
    return result


def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel, 0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_id, :]))) ** 2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference = inter_user_interference + (
                np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_index, :]))) ** 2
    denominator = noise_power + inter_user_interference

    result = numerator / denominator
    return result


def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)

    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * np.log2(1 + user_sinr)

    return result


def compute_sinr_nn(channel, precoder, noise_power, user_id, nr_of_users):
    numerator = tf.reduce_sum((tf.matmul(tf.transpose(channel[user_id]), precoder[user_id])) ** 2)
    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id:
            inter_user_interference = inter_user_interference + tf.reduce_sum(
                (tf.matmul(tf.transpose(channel[user_id]), precoder[user_index])) ** 2)
    denominator = noise_power + inter_user_interference

    result = numerator / denominator
    return result


def compute_WSR_nn(user_weights, channel, precoder, noise_power, nr_of_users):
    result = 0
    for batch_index in range(nr_of_samples_per_batch):
        for user_index in range(nr_of_users):
            user_sinr = compute_sinr_nn(channel[batch_index], precoder[batch_index], noise_power, user_index,
                                        nr_of_users)
            result = result + user_weights[batch_index][user_index] * (
                        tf.log(1 + user_sinr) / tf.log(tf.cast(2.0, tf.float64)))

    return result


# load a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.

def load_quadriga_channel(H_realization):
    channel_nn = []
    initial_transmitter_precoder = []
    for i_ue in range(nr_of_users):
        result_real = np.real(H_realization[i_ue]).reshape([-1,1])
        result_imag = np.imag(H_realization[i_ue]).reshape([-1,1])
        result_col_1 = np.vstack((result_real, result_imag))
        result_col_2 = np.vstack((-result_imag, result_real))
        result = np.hstack((result_col_1, result_col_2))
        initial_transmitter_precoder.append(result_col_1)
        channel_nn.append(result)

    initial_transmitter_precoder_array = np.array(initial_transmitter_precoder)
    ## 保证初始预编码值满足功率约束
    initial_transmitter_precoder_array1 = np.reshape(initial_transmitter_precoder_array,
                                                     [nr_of_users, 2, nr_of_BS_antennas])
    for i in range(nr_of_BSs):
        p_tmp = np.linalg.norm(initial_transmitter_precoder_array1[:, :, i * nr_of_antennas:(i + 1) * nr_of_antennas])
        if p_tmp > np.sqrt(total_power):
            initial_transmitter_precoder_array1[:, :,i * nr_of_antennas:(i + 1) * nr_of_antennas] = initial_transmitter_precoder_array1[:, :,
                                                           i * nr_of_antennas:(i + 1) * nr_of_antennas] / p_tmp * np.sqrt(total_power)
    initial_transmitter_precoder_array = np.reshape(initial_transmitter_precoder_array1, [nr_of_users, 2 * nr_of_BS_antennas, 1])
    initial_transmitter_precoder = []
    for i in range(nr_of_users):
        initial_transmitter_precoder.append(initial_transmitter_precoder_array[i])

    return channel_nn, initial_transmitter_precoder, H_realization


channel_realization_nn, init_transmitter_precoder, channel_complex = load_quadriga_channel(H[1])


# Builds one PGD iteration in the deep unfolded WMMSE network
def PGD_step(init, name, mse_weights, user_weights, receiver_precoder, channel, initial_transmitter_precoder,
             total_power):
    with tf.variable_scope(name):

        step_size = tf.Variable(tf.constant(init, dtype=tf.float64), name="step_size", dtype=tf.float64)
        # 梯度下降
        # First iteration
        a1_exp = tf.tile(tf.expand_dims(mse_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a2_exp = tf.tile(tf.expand_dims(user_weights[:, 0, :], -1), [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        # a3_exp = tf.tile(tf.expand_dims((tf.reduce_sum(receiver_precoder[:, 0, :, :], axis=-2)) ** 2, -1),
        #                  [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        a3_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:, 0, :, :])**2, axis=-2), -1),
                         [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
        temp = a1_exp * a2_exp * a3_exp * tf.matmul(channel[:, 0, :, :],
                                                    tf.transpose(channel[:, 0, :, :], perm=[0, 2, 1]))

        # Next iterations
        for i in range(1, nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1),
                             [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1),
                             [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            # a3_exp = tf.tile(tf.expand_dims((tf.reduce_sum(receiver_precoder[:, i, :, :], axis=-2)) ** 2, -1),
            #                  [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            a3_exp = tf.tile(tf.expand_dims(tf.reduce_sum((receiver_precoder[:, i, :, :]) ** 2, axis=-2), -1),
                             [1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas])
            temp = temp + a1_exp * a2_exp * a3_exp * tf.matmul(channel[:, i, :, :],
                                                               tf.transpose(channel[:, i, :, :], perm=[0, 2, 1]))
        sum_gradient = temp
        gradient = []

        # Gradient computation
        for i in range(nr_of_users):
            a1_exp = tf.tile(tf.expand_dims(mse_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            a2_exp = tf.tile(tf.expand_dims(user_weights[:, i, :], -1), [1, 2 * nr_of_BS_antennas, 1])
            gradient.append(step_size * (-2.0 * a1_exp * a2_exp * tf.matmul(channel[:, i, :, :],
                                                                            receiver_precoder[:, i, :,
                                                                            :]) + 2 * tf.matmul(sum_gradient,
                                                                                                initial_transmitter_precoder[
                                                                                                :, i, :, :])))
        tf.stack(gradient)
        gradient = tf.transpose(tf.stack(gradient), perm=[1, 0, 2, 3])
        output_temp = initial_transmitter_precoder - gradient

        # 投影
        output = []
        for i in range(nr_of_samples_per_batch): # projection
            initial_transmitter_precoder_array1 = tf.reshape(output_temp[i], [nr_of_users, 2, nr_of_BS_antennas])
            output1 =[]
            for ii in range(nr_of_BSs):
                output_temp_ue = initial_transmitter_precoder_array1[:, :, ii * nr_of_antennas:(ii + 1) * nr_of_antennas]
                p_tmp = tf.linalg.norm(output_temp_ue)
                output1.append(tf.cond(p_tmp ** 2 < total_power, lambda: initial_transmitter_precoder_array1[:, :, ii * nr_of_antennas:(ii + 1) * nr_of_antennas],
                                      lambda: tf.sqrt(tf.cast(total_power, tf.float64)) * initial_transmitter_precoder_array1[:, :, ii * nr_of_antennas:(ii + 1) * nr_of_antennas] / p_tmp))
            initial_transmitter_precoder_array1 = tf.stack(output1, axis=-2)
            output.append(tf.reshape(initial_transmitter_precoder_array1, [nr_of_users, 2 * nr_of_BS_antennas, 1]))

        # 强制稀疏
        xi = 2 #2
        output_sparse = []
        for i in range(nr_of_samples_per_batch):  # sparse
            initial_transmitter_precoder_array1 = tf.reshape(output[i],
                                                             [nr_of_users, 2, nr_of_BS_antennas])
            output1 = []
            for j in range(nr_of_users):
                output2 = []
                for ii in range(nr_of_BSs):
                    output_temp_ue = initial_transmitter_precoder_array1[j, :,
                                     ii * nr_of_antennas:(ii + 1) * nr_of_antennas]
                    p_tmp = tf.linalg.norm(output_temp_ue)
                    output2.append(tf.cond(p_tmp ** 2 < total_power / (xi * nr_of_users),
                                           lambda: tf.zeros([2, nr_of_antennas], dtype=tf.float64),
                                           lambda: output_temp_ue))
                output1.append(tf.stack(output2, axis=-2))

            initial_transmitter_precoder_array1 = tf.stack(output1)
            output_sparse.append(tf.reshape(initial_transmitter_precoder_array1, [nr_of_users, 2 * nr_of_BS_antennas, 1]))

        return tf.stack(output_sparse), step_size


def run_WMMSE(epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations,
              log=False):
    nr_of_users = np.size(channel, 0)
    nr_of_BS_antennas = np.size(channel, 1)
    WSR = []  # to check if the WSR (our cost function) increases at each iteration of the WMMSE
    break_condition = epsilon + 1  # break condition to stop the WMMSE iterations and exit the while
    receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(nr_of_users)  # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
    mse_weights = np.ones(nr_of_users)  # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
    transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users, nr_of_BS_antennas))  # transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)

    new_receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(nr_of_users)  # for the first iteration
    new_mse_weights = np.zeros(nr_of_users)  # for the first iteration
    new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros(
        (nr_of_users, nr_of_BS_antennas))  # for the first iteration

    # Initialization of transmitter precoder
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder[user_index, :] = channel[user_index, :]
    # 保证初始发射预编码矢量满足约束
    # transmitter_precoder = transmitter_precoder / np.linalg.norm(transmitter_precoder) * np.sqrt(total_power)
    for i in range(nr_of_BSs):
        p_tmp = np.linalg.norm(transmitter_precoder[:, i * nr_of_antennas:(i + 1) * nr_of_antennas])
        if p_tmp > np.sqrt(total_power):
            transmitter_precoder[:,i * nr_of_antennas:(i + 1) * nr_of_antennas] = transmitter_precoder[:,i * nr_of_antennas:(i + 1) * nr_of_antennas] / p_tmp * np.sqrt(total_power)

    # Store the WSR obtained with the initialized trasmitter precoder
    WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))

    # Compute the initial power of the transmitter precoder
    initial_power = 0
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index, :])) ** 2
    if log == True:
        print("Power of the initialized transmitter precoder:", initial_power)

    nr_of_iteration_counter = 0  # to keep track of the number of iteration of the WMMSE

    # while break_condition >= epsilon and nr_of_iteration_counter <= max_nr_of_iterations:
    while nr_of_iteration_counter <= max_nr_of_iterations:

        nr_of_iteration_counter = nr_of_iteration_counter + 1
        if log == True:
            print("WMMSE ITERATION: ", nr_of_iteration_counter)

        # Optimize receiver precoder - eq. (5) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0.0
                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1, :]),
                                                                transmitter_precoder[user_index_1, :]) / (
                                                                  noise_power + user_interference)

        # Optimize mse_weights - eq. (13) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0  # it includes the channel of all selected users
                inter_user_interference = 0  # it includes the channel of all selected users apart from the current one

                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2
                for user_index_2 in range(nr_of_users):
                    if user_index_2 != user_index_1 and user_index_2 in selected_users:
                        inter_user_interference = inter_user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_mse_weights[user_index_1] = (noise_power + user_interference) / (
                            noise_power + inter_user_interference)
        # A is J in SWMMSE
        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))
        D = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
                hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                               np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
                A = A + (new_mse_weights[user_index] * user_weights[user_index] * (
                    np.absolute(new_receiver_precoder[user_index])) ** 2) * hh
                D[user_index,:] = new_mse_weights[user_index] * user_weights[user_index] * channel[user_index, :] * new_receiver_precoder[user_index]

        I_q = 0
        I_Q = 5
        while I_q <= I_Q:
            I_q = I_q + 1
            for bs in range(nr_of_BSs):
                C = np.zeros((nr_of_users, nr_of_antennas)) + 1j*np.zeros((nr_of_users, nr_of_antennas))
                for user_index in selected_users:
                    tmp = np.zeros(nr_of_antennas)+1j*np.zeros(nr_of_antennas)
                    for bss in range(nr_of_BSs):
                        if bss != bs:
                            tmp = tmp + np.matmul(A[bs*nr_of_antennas:(bs+1)*nr_of_antennas, bss*nr_of_antennas:(bss+1)*nr_of_antennas], transmitter_precoder[user_index,bss*nr_of_antennas:(bss+1)*nr_of_antennas])
                    C[user_index,:] = D[user_index, bs*nr_of_antennas:(bs+1)*nr_of_antennas] - tmp

                C_max = np.max(np.sum(np.absolute(C)**2 , axis=1))
                mu_low = 0
                # mu_high = (total_power / nr_of_users)**(-0.5) * C_max
                mu_high = 100
                while mu_high - mu_low > 0.001:
                    mu_new = (mu_low + mu_high) / 2
                    tmp = np.linalg.inv(A[bs*nr_of_antennas:(bs+1)*nr_of_antennas, bs*nr_of_antennas:(bs+1)*nr_of_antennas] + mu_new*np.eye(nr_of_antennas))
                    tmp_precoder = np.matmul(tmp, np.transpose(C))
                    if np.sum(np.absolute(tmp_precoder)**2) > total_power:
                        mu_low = mu_new
                    else:
                        mu_high = mu_new
                tmp = np.linalg.inv(A[bs * nr_of_antennas:(bs + 1) * nr_of_antennas,
                                    bs * nr_of_antennas:(bs + 1) * nr_of_antennas] + mu_high * np.eye(nr_of_antennas))
                tmp_precoder = np.matmul(tmp, np.transpose(C))
                transmitter_precoder[:, bs*nr_of_antennas:(bs+1)*nr_of_antennas] = np.transpose(tmp_precoder)


        # To select only the weights of the selected users to check the break condition
        mse_weights_selected_users = []
        new_mse_weights_selected_users = []
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                mse_weights_selected_users.append(mse_weights[user_index])
                new_mse_weights_selected_users.append(new_mse_weights[user_index])

        mse_weights = deepcopy(new_mse_weights)
        # transmitter_precoder = deepcopy(new_transmitter_precoder)
        receiver_precoder = deepcopy(new_receiver_precoder)

        WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))
        break_condition = np.absolute(
            np.log2(np.prod(new_mse_weights_selected_users)) - np.log2(np.prod(mse_weights_selected_users)))

    if log == True:
        plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
        plt.plot(WSR, 'bo')
        plt.show()

    return transmitter_precoder, receiver_precoder, mse_weights, WSR


tf.reset_default_graph()  #tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形。

channel_input = tf.placeholder(tf.float64, shape=None, name='channel_input')
initial_tp = tf.placeholder(tf.float64, shape=None, name='initial_transmitter_precoder')
channel_complex_input = tf.placeholder(tf.complex64, shape=None, name='channel_complex_input')
initial_transmitter_precoder = initial_tp

# debug
# channel_input = []
# initial_transmitter_precoder = []
# # Building a batch for training
# for ii in range(nr_of_samples_per_batch):
#     channel_realization_nn, init_transmitter_precoder_tmp, channel_complex = load_quadriga_channel(H[ii])
#     channel_input.append(channel_realization_nn)
#     initial_transmitter_precoder.append(init_transmitter_precoder_tmp)
#
# channel_input = np.array(channel_input)
# initial_transmitter_precoder = np.array(initial_transmitter_precoder)


# Arrays that contain the initialization values of the step sizes.
# The number of step sizes depends on the selected number of PGD layers, the number of elements for each step size initializer depends on the selected number of deep unfolded iterations
step_size1_init = [1.0] * nr_of_iterations_nn
step_size2_init = [1.0] * nr_of_iterations_nn
step_size3_init = [1.0] * nr_of_iterations_nn
step_size4_init = [1.0] * nr_of_iterations_nn

# Used to collect the step sizes at each iteration
all_step_size1_temp = []
all_step_size2_temp = []
all_step_size3_temp = []
all_step_size4_temp = []

profit = []  # stores the WSR obtained at each iteration

for loop in range(0, nr_of_iterations_nn):

    user_interference2 = []
    for batch_index in range(nr_of_samples_per_batch):
        user_interference_single = []
        for i in range(nr_of_users):
            temp = 0.0
            for j in range(nr_of_users):
                temp = temp + tf.reduce_sum((tf.matmul(tf.transpose(channel_input[batch_index, i, :, :]),
                                                       initial_transmitter_precoder[batch_index, j, :, :])) ** 2)
            user_interference_single.append(temp + noise_power)
        user_interference2.append(user_interference_single)

    tf.stack(user_interference2)

    user_interference_exp2 = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(user_interference2, -1), [1, 1, 2]), -1),
                                     [1, 1, 1, 1])

    receiver_precoder_temp = (tf.matmul(tf.transpose(channel_input, perm=[0, 1, 3, 2]), initial_transmitter_precoder))
    # Optimize the receiver precoder
    receiver_precoder = tf.divide(receiver_precoder_temp, user_interference_exp2)

    # Optimize the mmse weights
    self_interference = tf.reduce_sum(
        (tf.matmul(tf.transpose(channel_input, perm=[0, 1, 3, 2]), initial_transmitter_precoder)) ** 2, axis=2)

    inter_user_interference_total = []

    for batch_index in range(nr_of_samples_per_batch):
        inter_user_interference_temp = []
        for i in range(nr_of_users):
            temp = 0.0
            for j in range(nr_of_users):
                if j != i:
                    temp = temp + tf.reduce_sum((tf.matmul(tf.transpose(channel_input[batch_index, i, :, :]),
                                                           initial_transmitter_precoder[batch_index, j, :, :])) ** 2)
            inter_user_interference_temp.append(temp + noise_power)  # $sum{|(h_i)*H,v_i}|**2 + noise_power$
        inter_user_interference = tf.reshape(tf.stack(inter_user_interference_temp),
                                             (nr_of_users, 1))  # Nx1 $sum{|(h_i)*H,v_i}|**2 + noise_power$
        inter_user_interference_total.append(inter_user_interference)

    mse_weights = (tf.divide(self_interference, inter_user_interference_total)) + 1.0

    # Optimize the transmitter precoder through PGD
    transmitter_precoder1, step_size1 = PGD_step(step_size1_init[loop], 'PGD_step1', mse_weights, user_weights,
                                                 receiver_precoder, channel_input, initial_transmitter_precoder,
                                                 total_power)

    transmitter_precoder2, step_size2 = PGD_step(step_size2_init[loop], 'PGD_step2', mse_weights, user_weights,
                                                 receiver_precoder, channel_input, transmitter_precoder1, total_power)
    transmitter_precoder3, step_size3 = PGD_step(step_size3_init[loop], 'PGD_step3', mse_weights, user_weights,
                                                 receiver_precoder, channel_input, transmitter_precoder2, total_power)
    transmitter_precoder, step_size4 = PGD_step(step_size4_init[loop], 'PGD_step4', mse_weights, user_weights,
                                                receiver_precoder, channel_input, transmitter_precoder3, total_power)

    initial_transmitter_precoder = transmitter_precoder
    all_step_size1_temp.append(step_size1)
    all_step_size2_temp.append(step_size2)
    all_step_size3_temp.append(step_size3)
    all_step_size4_temp.append(step_size4)

    # The WSR achieved with the transmitter precoder obtained at the current iteration is appended
    profit.append(compute_WSR_nn(user_weights, channel_input, initial_transmitter_precoder, noise_power, nr_of_users))

all_step_size1 = tf.stack(all_step_size1_temp)
all_step_size2 = tf.stack(all_step_size2_temp)
all_step_size3 = tf.stack(all_step_size3_temp)
all_step_size4 = tf.stack(all_step_size4_temp)

final_precoder = initial_transmitter_precoder  # this is the last transmitter precoder, i.e. the one that will be actually used for transmission

WSR = tf.reduce_sum(tf.stack(profit))  # this is the cost function to maximize, i.e. the WSR obtained if we use the transmitter precoder that we have at each round of the loop
WSR_iteration = tf.reduce_mean(tf.stack(profit, axis=1), axis=0) / nr_of_samples_per_batch
WSR_final = compute_WSR_nn(user_weights, channel_input, final_precoder, noise_power,
                           nr_of_users) / nr_of_samples_per_batch  # this is the WSR computed using the "final_precoder"

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(-WSR)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(-WSR)

WSR_WMMSE = []  # to store the WSR attained by the WMMSE
WSR_ZF = []  # to store the WSR attained by the zero-forcing
WSR_RZF = []  # to store the WSR attained by the regularized zero-forcing
WSR_nn = []  # to store the WSR attained by the deep unfolded WMMSE
WSR_nn1 = []  # to store the WSR attained by the deep unfolded WMMSE
training_loss = []
tx_precoder = []

with tf.Session() as sess:
    print("start of session")
    start_of_time = time.time()
    # print('channel')
    sess.run(tf.global_variables_initializer())
    # print('channel')
    for i in range(nr_of_batches_training):
        batch_for_training = []
        initial_transmitter_precoder_batch = []
        # Building a batch for training
        for ii in range(nr_of_samples_per_batch):
            channel_realization_nn, init_transmitter_precoder, channel_complex = load_quadriga_channel(H[i*nr_of_samples_per_batch+ii])
            batch_for_training.append(channel_realization_nn)
            initial_transmitter_precoder_batch.append(init_transmitter_precoder)
        sess.run(optimizer,
                 feed_dict={channel_input: batch_for_training, initial_tp: initial_transmitter_precoder_batch})
        training_loss.append(-1 * (sess.run(WSR, feed_dict={channel_input: batch_for_training,
                                                            initial_tp: initial_transmitter_precoder_batch
                                                        })))

        print(i, training_loss[-1])

    print("step size1", sess.run(all_step_size1))
    print("step size2", sess.run(all_step_size2))
    print("step size3", sess.run(all_step_size3))
    print("step size4", sess.run(all_step_size4))

    print("Training took:", time.time() - start_of_time)

    # path = 'training_loss_220505.mat'
    # sio.savemat(path, {'training_loss': training_loss})

    # For repeatability, test
    np.random.seed(1234)
    for i in range(nr_of_batches_test):
        batch_for_testing = []
        initial_transmitter_precoder_batch = []
        WSR_WMMSE_batch = np.zeros([1, nr_of_iterations])
        WSR_ZF_batch = 0.0
        WSR_RZF_batch = 0.0

        # Building a batch for testing
        for ii in range(nr_of_samples_per_batch):
            channel_realization_nn, init_transmitter_precoder, channel_realization_regular = load_quadriga_channel(H_test[i*nr_of_samples_per_batch+ii])

            # Compute the WMMSE solution
            _, _, _, WSR_WMMSE_one_sample = run_WMMSE(epsilon, channel_realization_regular, scheduled_users,
                                                      total_power, noise_power, user_weights_for_regular_WMMSE,
                                                      nr_of_iterations - 1, log=False)


            WSR_WMMSE_batch = WSR_WMMSE_batch + WSR_WMMSE_one_sample[1:]
            batch_for_testing.append(channel_realization_nn)
            initial_transmitter_precoder_batch.append(init_transmitter_precoder)

        # Testing
        WSR_nn1.append(sess.run(WSR_final, feed_dict={channel_input: batch_for_testing,
                                                     initial_tp: initial_transmitter_precoder_batch}))
        # WSR_nn.append(sess.run(WSR_iteration, feed_dict={channel_input: batch_for_testing,
        #                                              initial_tp: initial_transmitter_precoder_batch}))
        WSR_WMMSE.append(WSR_WMMSE_batch / nr_of_samples_per_batch)
        tx_precoder.append(sess.run(final_precoder, feed_dict={channel_input: batch_for_testing,
                                                     initial_tp: initial_transmitter_precoder_batch}))

print("Training and testing took:", time.time() - start_of_time)
print("The WSR acheived with the deep unfolded WMMSE algorithm is: ", np.mean(WSR_nn1))
# print("The WSR acheived with the deep unfolded WMMSE algorithm is: ", np.mean(np.array(WSR_nn), axis=0))
print("The WSR acheived with the WMMSE algorithm is: ", np.mean(WSR_WMMSE, axis=0))
print("Average number of serving bss per user: ", np.count_nonzero(np.array(tx_precoder)))

step_size_hold =[[],[],[],[]]
step_size_hold[0]=all_step_size1
step_size_hold[1]=all_step_size2
step_size_hold[2]=all_step_size3
step_size_hold[3]=all_step_size4

path = 'training_loss.mat'
sio.savemat(path, {'training_loss': training_loss, 'test_unfolding': WSR_nn, 'test_WMMSE': WSR_WMMSE, 'stepsize': step_size_hold})

plt.figure()
plt.plot(training_loss)
plt.ylabel("Training loss")
plt.xlabel("Sample index")
plt.show()
plt.savefig("loss.png")