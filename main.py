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

###
# Common hyperparameters
weight_decay = 1e-4
batch_size = 10
suffix = model_name
lr_decay_per_round = 1
sch_step = 1
sch_gamma = 1

# Model function
model_func = lambda: combined_model(model_name)
init_model = model_func()

# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(17)
init_model = model_func()
if not os.path.exists('%s/Model/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)):
    if not os.path.exists('%s/Model/%s/' % (data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%s/Model/%s/' % (data_path, data_obj.name))
    torch.save(init_model.state_dict(), '%s/Model/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%s/Model/%s/%s_init_mdl.pt' % (data_path, data_obj.name, model_name)))

save_tasks = False
trial = False
train_all = 1
learning_rate = .1
K_list = [10, 20]

alpha_list = [1, 5, 10]
num_grad_step_list = [1,5]
learning_rate_ft_list = [0.1,0.01]


save_models = True
save_performance = True
save_tensorboard = True


### Methods
print('Train FS')
taskLrFT_GS = [[0.1, 1], [0.1, 5]]
add_noFt = True
for K in K_list:
    print_per = K // 4 if K > 4 else 1
    _ = train_FS(data_obj=data_obj, learning_rate=learning_rate, batch_size=batch_size, K=K, print_per=print_per,
                  weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                  taskLrFT_GS=taskLrFT_GS, add_noFt=add_noFt,
                  sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                  save_models=save_models, save_performance=save_performance, save_tensorboard=save_tensorboard,
                  suffix=suffix, data_path=data_path)


print('Train TOE')
taskLrFT_GS = [[0.1, 1], [0.1, 5]]
add_noFt = True
for K in K_list:
    print_per = K // 4 if K > 4 else 1
    _ = train_TOE(data_obj=data_obj, learning_rate=learning_rate, batch_size=batch_size, K=K, print_per=print_per,
                  weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                  taskLrFT_GS=taskLrFT_GS, add_noFt=add_noFt,
                  sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                  save_models=save_models, save_performance=save_performance, save_tensorboard=save_tensorboard,
                  suffix=suffix, data_path=data_path)


print('Train FTML')
for K in K_list:
    for learning_rate_ft in learning_rate_ft_list:
        for num_grad_step in num_grad_step_list:
            print('K %3d, GS %3d' % (K, num_grad_step))
            print_per = K // 4 if K > 4 else 1
            _ = train_FTML(data_obj=data_obj, learning_rate=learning_rate, learning_rate_ft=learning_rate_ft,
                           batch_size=batch_size, K=K, num_grad_step=num_grad_step, print_per=print_per,
                           weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                           sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                           save_models=save_models, save_performance=save_performance, save_tensorboard=save_tensorboard,
                           suffix=suffix, data_path=data_path)

print('Train MOGD')
for K in K_list:
    for learning_rate_ft in learning_rate_ft_list:
        for num_grad_step in num_grad_step_list:
            print('K %3d, GS %3d' % (K, num_grad_step))
            print_per = K // 4 if K > 4 else 1
            _ = train_MOGD(data_obj=data_obj, learning_rate=learning_rate, learning_rate_ft=learning_rate_ft,
                           batch_size=batch_size, K=K, num_grad_step=num_grad_step, print_per=print_per,
                           weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                           sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                           save_models=save_models, save_performance=save_performance, save_tensorboard=save_tensorboard,
                           suffix=suffix, data_path=data_path)

print('Train MOML')
for K in K_list:
    for num_grad_step in num_grad_step_list:
        for alpha in alpha_list:
            print('K %3d, GS %3d, alpha %f' % (K, num_grad_step, alpha))
            print_per = K // 4 if K > 4 else 1
            _ = train_MOML(data_obj=data_obj, alpha=alpha, learning_rate=learning_rate,
                           learning_rate_ft=learning_rate_ft,
                           batch_size=batch_size, K=K, num_grad_step=num_grad_step, print_per=print_per,
                           weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                           sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                           save_models=save_models, save_performance=save_performance,
                           save_tensorboard=save_tensorboard,
                           suffix=suffix, data_path=data_path)



buffer_size=10
buffer_style='Random' # choose of Random or Last
for K in K_list:
    for learning_rate_ft in learning_rate_ft_list:
        for num_grad_step in num_grad_step_list:
            for alpha in alpha_list:
                print('K %3d, GS %3d, alpha %f' % (K, num_grad_step, alpha))
                print_per = K // 4 if K > 4 else 1
                _ = train_BMOML(data_obj=data_obj, alpha=alpha, learning_rate=learning_rate,
                               learning_rate_ft=learning_rate_ft,
                               batch_size=batch_size, K=K, num_grad_step=num_grad_step, print_per=print_per,
                               weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                               sch_step=sch_step, sch_gamma=sch_gamma, lr_decay_per_round=lr_decay_per_round,
                               save_models=save_models, save_performance=save_performance,
                               save_tensorboard=save_tensorboard, buffer_size=buffer_size, style=buffer_style,
                               suffix=suffix, data_path=data_path)