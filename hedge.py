from utils_methods import *
# Dataset initialization
# You can change different dataset for CIFAR100 and miniImageNet
# When using CIFAR100 and miniImageNet, you also need to change rule='way'
# To replcate the results, number of task n_task and model_name need to change accordingly
data_path = '/home/rzhu/code/MOMLclean/MNIST1000_clean'  # The folder to save Data & Model MNIST1000_v1_new
rule = 'mnist_rotations_flips_crop_scale'
seed = 17
n_task = 1000
data_obj = DatasetObject(dataset='mnist', seed=seed, n_task=n_task, rule=rule, data_path=data_path)
model_name = 'mnist_2NN'  # Model type

# parameter configurations for experts
#common used
weight_decay = 1e-4
batch_size = 10
lr_decay_per_round = 1
sch_step = 1
sch_gamma = 1

num_grad_step_list = [1,5]
alpha_list = [1,5,10]
K_list = [10,20]
learning_rate_ft_list = [0.01,0.1]
learning_rate = .1

CTM_MOML = []
LTM_MOML = []
model_dir = '/home/rzhu/code/MOMLclean/MNIST1000_clean/Model/mnist_17_mnist_rotations_flips_crop_scale_1000/'
suffix = 'MOML_' + model_name
CTM_dir = '/1000_CTM_perf.npy'
LTM_dir = '/1000_LTM_perf.npy'

# load saved loss of all expertes
for K in K_list:
    for learning_rate_ft in learning_rate_ft_list:
        for num_grad_step in num_grad_step_list:
            for alpha in alpha_list:
                MOML_dir = suffix +'_alpha%f_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f_nomo' % (alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay)
                MOML_dir += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)
                MOML_dir = model_dir+MOML_dir
                CTM_MOML.append(np.load(MOML_dir+CTM_dir))
                LTM_MOML.append(np.load(MOML_dir+LTM_dir))

# Hedge Computation of CTM
expert_n = len(K_list) * len(learning_rate_ft_list) * len(alpha_list)
w = np.ones((expert_n,1))
w = w/np.sum(w)
print(w)
beta = 10

MOML_CTM = []
MOML_CTM_avg =[]
for t in range(n_task):
    acc = 0
    for i in range(expert_n):
        #loss_i =  CTM[i][t][0]
        loss_i =  1-CTM_MOML[i][t][1]
        w[i] = w[i] * np.power(beta,-loss_i)
    w = w/np.sum(w)
    for i in range(expert_n):
        acc+= w[i]*CTM_MOML[i][t][1]
    MOML_CTM.append(acc)
    MOML_CTM_avg.append(sum(MOML_CTM) / len(MOML_CTM))

experts_avg = []
for i in range(expert_n):
    expert_avg = []
    for t in range(n_task):
        expert_avg.append(np.mean(CTM_MOML[i][:t+1,1]))
    experts_avg.append(expert_avg)

# Hedge Computation of LTM
w = np.ones((expert_n,1))
w = w/np.sum(w)
print(w)
beta = 10

MOML_LTM = []
MOML_LTM_avg =[]
for t in range(n_task):
    acc = 0
    for i in range(expert_n):
        #loss_i =  LTM[i][t][0]
        loss_i =  1-LTM_MOML[i][t][1]
        w[i] = w[i] * np.power(beta,-loss_i)
    w = w/np.sum(w)
    for i in range(expert_n):
        acc+= w[i]*LTM_MOML[i][t][1]
    MOML_LTM.append(acc)
    MOML_LTM_avg.append(sum(MOML_LTM) / len(MOML_LTM))

np.save(model_dir + '_CTM_Hedge.npy', MOML_CTM_avg)
np.save(model_dir + '_CTM_Hedge.npy', MOML_LTM)