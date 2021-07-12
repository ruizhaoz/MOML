from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *


### Methods
def train_FS(data_obj, learning_rate, batch_size, K, print_per, weight_decay, model_func,
              taskLrFT_GS, add_noFt, init_model, sch_step, sch_gamma, lr_decay_per_round,
              save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'FS_' + suffix
    suffix += '_Lr%f_B%d_K%d_W%f_fixed' % (learning_rate, batch_size, K, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x;
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks

    len_ = len(taskLrFT_GS) + 1 if add_noFt else len(taskLrFT_GS)

    tst_after_perf = np.zeros((len_, n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((len_, n_tasks, 2))

    CTM_perf = np.zeros((len_, n_tasks, 2))
    LTM_perf = np.zeros((len_, n_tasks, 2))

    if save_tensorboard:
        writer_list = []
        for [learning_rate_ft, num_grad_step] in taskLrFT_GS:
            writer_list.append(SummaryWriter(
                '%s/Runs/%s/%s_LrT%f_GS%d' % (data_path, data_obj.name, suffix, learning_rate_ft, num_grad_step)))
        if add_noFt:
            writer_list.append(SummaryWriter('%s/Runs/%s/%s_noFT' % (data_path, data_obj.name, suffix)))

    if not os.path.exists('%s/Model/%s/%s/%d_tst_after_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Each round is trained on previous all tasks
        for task in range(n_tasks):
            trn_x = np.concatenate(task_x[:task + 1]);

            print('---- Round %2d, Total datapoints: %5d' % (task + 1, len(trn_x)))

            # Start from the initial model
            _model = model_func().to(device)
            _model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            ### Evaluation of TOE is a bit different
            for ii, [learning_rate_ft, num_grad_step] in enumerate(taskLrFT_GS):
                CTM_perf[ii, task] = get_maml_acc_loss(task_x[task], task_y[task], _model, model_func,
                                                       learning_rate_ft, num_grad_step, dataset_name,
                                                       tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            if add_noFt:
                CTM_perf[-1, task] = get_acc_loss(data_obj.tst_x[task], data_obj.tst_y[task], _model, dataset_name)

            ### Evaluation
            for ii, [learning_rate_ft, num_grad_step] in enumerate(taskLrFT_GS):
                print('Lr Task: %f, GS: %d' % (learning_rate_ft, num_grad_step))
                trn_after_perf[ii, task] = get_maml_acc_loss(task_x[task], task_y[task], _model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name)
                for tt in range(task + 1):
                    tst_after_perf[ii, task, tt] = get_maml_acc_loss(task_x[tt], task_y[tt], _model, model_func,
                                                                     learning_rate_ft, num_grad_step, dataset_name,
                                                                     tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])
                # LTM
                LTM_perf[ii, task] = np.mean(tst_after_perf[ii, task, :task + 1, :], axis=0)
                print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                      % (task + 1, trn_after_perf[ii, task][1], trn_after_perf[ii, task][0]))
                print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                      % (task + 1, tst_after_perf[ii, task, task][1], tst_after_perf[ii, task, task][0]))

            if add_noFt:
                print('No FT')
                trn_after_perf[-1, task] = get_acc_loss(task_x[task], task_y[task], _model, dataset_name)
                for tt in range(task + 1):
                    tst_after_perf[-1, task, tt] = get_acc_loss(data_obj.tst_x[tt], data_obj.tst_y[tt], _model,
                                                                dataset_name)
                print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                      % (task + 1, trn_after_perf[-1, task][1], trn_after_perf[-1, task][0]))
                print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                      % (task + 1, tst_after_perf[-1, task, task][1], tst_after_perf[-1, task, task][0]))
                LTM_perf[-1, task] = np.mean(tst_after_perf[-1, task, :task + 1, :], axis=0)

            if save_tensorboard:
                for idx_ in range(len(writer_list)):
                    ## Loss
                    writer_list[idx_].add_scalar('Loss/Train_After', trn_after_perf[idx_, task, 0], task + 1)

                    writer_list[idx_].add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[idx_, :task + 1, 0]),
                                                 task + 1)

                    writer_list[idx_].add_scalar('Loss/CTM', CTM_perf[idx_, task, 0], task + 1)
                    writer_list[idx_].add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[idx_, :task + 1, 0]), task + 1)

                    writer_list[idx_].add_scalar('Loss/LTM', LTM_perf[idx_, task, 0], task + 1)
                    writer_list[idx_].add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[idx_, :task + 1, 0]), task + 1)

                    writer_list[idx_].add_scalar('Loss/Task_1', tst_after_perf[idx_, task, 0, 0], task + 1)

                    ## Accuracy
                    writer_list[idx_].add_scalar('Accuracy/Train_After', trn_after_perf[idx_, task, 1], task + 1)

                    writer_list[idx_].add_scalar('Accuracy/Train_After_Avg',
                                                 np.mean(trn_after_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/CTM', CTM_perf[idx_, task, 1], task + 1)
                    writer_list[idx_].add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/LTM', LTM_perf[idx_, task, 1], task + 1)
                    writer_list[idx_].add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/Task_1', tst_after_perf[idx_, task, 0, 1], task + 1)

            online_mdls[task] = _model

            if save_models:
                torch.save(_model.state_dict(), '%s/Model/%s/%s/%d_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)
    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                _model = model_func().to(device)
                _model.load_state_dict(torch.load('%s/Model/%s/%s/%d_model.pt'
                                                  % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = _model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf





def train_TOE(data_obj, learning_rate, batch_size, K, print_per, weight_decay, model_func,
              taskLrFT_GS, add_noFt, init_model, sch_step, sch_gamma, lr_decay_per_round,
              save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'TOE_' + suffix
    suffix += '_Lr%f_B%d_K%d_W%f_fixed' % (learning_rate, batch_size, K, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x;
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks

    len_ = len(taskLrFT_GS) + 1 if add_noFt else len(taskLrFT_GS)

    tst_after_perf = np.zeros((len_, n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((len_, n_tasks, 2))

    CTM_perf = np.zeros((len_, n_tasks, 2))
    LTM_perf = np.zeros((len_, n_tasks, 2))

    if save_tensorboard:
        writer_list = []
        for [learning_rate_ft, num_grad_step] in taskLrFT_GS:
            writer_list.append(SummaryWriter(
                '%s/Runs/%s/%s_LrT%f_GS%d' % (data_path, data_obj.name, suffix, learning_rate_ft, num_grad_step)))
        if add_noFt:
            writer_list.append(SummaryWriter('%s/Runs/%s/%s_noFT' % (data_path, data_obj.name, suffix)))

    if not os.path.exists('%s/Model/%s/%s/%d_tst_after_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Each round is trained on previous all tasks
        for task in range(n_tasks):
            trn_x = np.concatenate(task_x[:task + 1]);
            trn_y = np.concatenate(task_y[:task + 1])  # train on all previous tasks
            tst_x = False;
            tst_y = False
            print('---- Round %2d, Total datapoints: %5d' % (task + 1, len(trn_x)))

            # Start from the initial model
            _model = model_func().to(device)
            _model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            decay = lr_decay_per_round ** task
            _model = train_model(_model, trn_x, trn_y, tst_x, tst_y, learning_rate * decay, batch_size, K,
                                 print_per, weight_decay, dataset_name, sch_step, sch_gamma)

            ### Evaluation of TOE is a bit different
            for ii, [learning_rate_ft, num_grad_step] in enumerate(taskLrFT_GS):
                CTM_perf[ii, task] = get_maml_acc_loss(task_x[task], task_y[task], _model, model_func,
                                                       learning_rate_ft, num_grad_step, dataset_name,
                                                       tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            if add_noFt:
                CTM_perf[-1, task] = get_acc_loss(data_obj.tst_x[task], data_obj.tst_y[task], _model, dataset_name)

            ### Evaluation
            for ii, [learning_rate_ft, num_grad_step] in enumerate(taskLrFT_GS):
                print('Lr Task: %f, GS: %d' % (learning_rate_ft, num_grad_step))
                trn_after_perf[ii, task] = get_maml_acc_loss(task_x[task], task_y[task], _model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name)
                for tt in range(task + 1):
                    tst_after_perf[ii, task, tt] = get_maml_acc_loss(task_x[tt], task_y[tt], _model, model_func,
                                                                     learning_rate_ft, num_grad_step, dataset_name,
                                                                     tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])
                # LTM
                LTM_perf[ii, task] = np.mean(tst_after_perf[ii, task, :task + 1, :], axis=0)
                print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                      % (task + 1, trn_after_perf[ii, task][1], trn_after_perf[ii, task][0]))
                print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                      % (task + 1, tst_after_perf[ii, task, task][1], tst_after_perf[ii, task, task][0]))

            if add_noFt:
                print('No FT')
                trn_after_perf[-1, task] = get_acc_loss(task_x[task], task_y[task], _model, dataset_name)
                for tt in range(task + 1):
                    tst_after_perf[-1, task, tt] = get_acc_loss(data_obj.tst_x[tt], data_obj.tst_y[tt], _model,
                                                                dataset_name)
                print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                      % (task + 1, trn_after_perf[-1, task][1], trn_after_perf[-1, task][0]))
                print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                      % (task + 1, tst_after_perf[-1, task, task][1], tst_after_perf[-1, task, task][0]))
                LTM_perf[-1, task] = np.mean(tst_after_perf[-1, task, :task + 1, :], axis=0)

            if save_tensorboard:
                for idx_ in range(len(writer_list)):
                    ## Loss
                    writer_list[idx_].add_scalar('Loss/Train_After', trn_after_perf[idx_, task, 0], task + 1)

                    writer_list[idx_].add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[idx_, :task + 1, 0]),
                                                 task + 1)

                    writer_list[idx_].add_scalar('Loss/CTM', CTM_perf[idx_, task, 0], task + 1)
                    writer_list[idx_].add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[idx_, :task + 1, 0]), task + 1)

                    writer_list[idx_].add_scalar('Loss/LTM', LTM_perf[idx_, task, 0], task + 1)
                    writer_list[idx_].add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[idx_, :task + 1, 0]), task + 1)

                    writer_list[idx_].add_scalar('Loss/Task_1', tst_after_perf[idx_, task, 0, 0], task + 1)

                    ## Accuracy
                    writer_list[idx_].add_scalar('Accuracy/Train_After', trn_after_perf[idx_, task, 1], task + 1)

                    writer_list[idx_].add_scalar('Accuracy/Train_After_Avg',
                                                 np.mean(trn_after_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/CTM', CTM_perf[idx_, task, 1], task + 1)
                    writer_list[idx_].add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/LTM', LTM_perf[idx_, task, 1], task + 1)
                    writer_list[idx_].add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[idx_, :task + 1, 1]), task + 1)

                    writer_list[idx_].add_scalar('Accuracy/Task_1', tst_after_perf[idx_, task, 0, 1], task + 1)

            online_mdls[task] = _model

            if save_models:
                torch.save(_model.state_dict(), '%s/Model/%s/%s/%d_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)
    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                _model = model_func().to(device)
                _model.load_state_dict(torch.load('%s/Model/%s/%s/%d_model.pt'
                                                  % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = _model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf


def train_FTML(data_obj, learning_rate, learning_rate_ft, batch_size, K, num_grad_step,
               print_per, weight_decay, model_func, init_model, sch_step, sch_gamma, lr_decay_per_round,
               save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'FTML_' + suffix
    suffix += '_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f_nomo' % (
    learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks
    writer = SummaryWriter('%s/Runs/%s/%s' % (data_path, data_obj.name, suffix)) if save_tensorboard else None

    tst_before_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_before_perf = np.zeros((n_tasks, 2))

    tst_after_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((n_tasks, 2))

    CTM_perf = np.zeros((n_tasks, 2))
    LTM_perf = np.zeros((n_tasks, 2))

    # Initialize model
    meta_model = model_func().to(device)
    meta_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())), strict=False)

    if not os.path.exists('%s/Model/%s/%s/%d_tst_before_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Train
        for task in range(n_tasks):
            print('---- Round %2d' % (task + 1))

            ### Evaluation
            # Test all seen tasks including the current one before training.
            for tt in range(task):
                tst_before_perf[task][tt] = tst_after_perf[task - 1][
                    tt]  # The model is not updated, no need to calculate twice

            tst_before_perf[task][task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                            learning_rate_ft, num_grad_step, dataset_name,
                                                            tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            trn_before_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                      learning_rate_ft, num_grad_step, dataset_name)

            # Train
            trn_x = task_x[:task + 1]
            trn_y = task_y[:task + 1]
            decay = lr_decay_per_round ** task
            meta_model = train_FTML_model(meta_model, model_func, trn_x, trn_y, learning_rate * decay, learning_rate_ft,
                                          num_grad_step, batch_size, K, print_per, weight_decay, dataset_name,
                                          sch_step, sch_gamma)

            ### Evaluation
            trn_after_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                     learning_rate_ft, num_grad_step, dataset_name)

            # Test all seen tasks.
            for tt in range(task + 1):
                tst_after_perf[task][tt] = get_maml_acc_loss(task_x[tt], task_y[tt], meta_model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name,
                                                             tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])

            ### CTM and LTM
            CTM_perf[task] = tst_before_perf[task][task]
            LTM_perf[task] = np.mean(tst_after_perf[task, :task + 1, :], axis=0)

            print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                  % (task + 1, trn_after_perf[task][1], trn_after_perf[task][0]))
            print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                  % (task + 1, tst_after_perf[task, task, 1], tst_after_perf[task, task, 0]))

            if save_tensorboard:
                ## Loss
                writer.add_scalar('Loss/Train_Before', trn_before_perf[task, 0], task + 1)
                writer.add_scalar('Loss/Train_After', trn_after_perf[task, 0], task + 1)

                writer.add_scalar('Loss/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 0]), task + 1)
                writer.add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/CTM', CTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/LTM', LTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/Task_1', tst_after_perf[task, 0, 0], task + 1)

                ## Accuracy
                writer.add_scalar('Accuracy/Train_Before', trn_before_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/Train_After', trn_after_perf[task, 1], task + 1)

                writer.add_scalar('Accuracy/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 1]), task + 1)
                writer.add_scalar('Accuracy/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/CTM', CTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/LTM', LTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/Task_1', tst_after_perf[task, 0, 1], task + 1)

            online_mdls[task] = meta_model

            if save_models:
                torch.save(meta_model.state_dict(), '%s/Model/%s/%s/%d_meta_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
            np.save(path_ + '_tst_before_perf.npy', tst_before_perf)
            np.save(path_ + '_trn_before_perf.npy', trn_before_perf)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)


    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                meta_model = model_func().to(device)
                meta_model.load_state_dict(torch.load('%s/Model/%s/%s/%d_meta_model.pt'
                                                      % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = meta_model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
        tst_before_perf = np.load(path_ + '_tst_before_perf.npy')
        trn_before_perf = np.load(path_ + '_trn_before_perf.npy')

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_before_perf, trn_before_perf, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf


def train_MOGD(data_obj, learning_rate, learning_rate_ft, batch_size, K, num_grad_step,
               print_per, weight_decay, model_func, init_model, sch_step, sch_gamma, lr_decay_per_round,
               save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'MOGD_' + suffix
    suffix += '_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f_nomo' % (
    learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks
    writer = SummaryWriter('%s/Runs/%s/%s' % (data_path, data_obj.name, suffix)) if save_tensorboard else None

    tst_before_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_before_perf = np.zeros((n_tasks, 2))

    tst_after_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((n_tasks, 2))

    CTM_perf = np.zeros((n_tasks, 2))
    LTM_perf = np.zeros((n_tasks, 2))

    # Initialize model
    meta_model = model_func().to(device)
    meta_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())), strict=False)

    if not os.path.exists('%s/Model/%s/%s/%d_tst_before_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Train
        for task in range(n_tasks):
            print('---- Round %2d' % (task + 1))

            ### Evaluation
            # Test all seen tasks including the current one before training.
            for tt in range(task):
                tst_before_perf[task][tt] = tst_after_perf[task - 1][
                    tt]  # The model is not updated, no need to calculate twice

            tst_before_perf[task][task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                            learning_rate_ft, num_grad_step, dataset_name,
                                                            tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            trn_before_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                      learning_rate_ft, num_grad_step, dataset_name)

            # Train only get the current task
            trn_x = task_x[task]
            trn_y = task_y[task]
            decay = lr_decay_per_round ** task
            meta_model = train_MOGD_model(meta_model, model_func, trn_x, trn_y, learning_rate * decay, learning_rate_ft,
                                          num_grad_step, batch_size, K, print_per, weight_decay, dataset_name,
                                          sch_step, sch_gamma)

            ### Evaluation
            trn_after_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                     learning_rate_ft, num_grad_step, dataset_name)

            # Test all seen tasks.
            for tt in range(task + 1):
                tst_after_perf[task][tt] = get_maml_acc_loss(task_x[tt], task_y[tt], meta_model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name,
                                                             tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])

            ### CTM and LTM
            CTM_perf[task] = tst_before_perf[task][task]
            LTM_perf[task] = np.mean(tst_after_perf[task, :task + 1, :], axis=0)

            print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                  % (task + 1, trn_after_perf[task][1], trn_after_perf[task][0]))
            print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                  % (task + 1, tst_after_perf[task, task, 1], tst_after_perf[task, task, 0]))

            if save_tensorboard:
                ## Loss
                writer.add_scalar('Loss/Train_Before', trn_before_perf[task, 0], task + 1)
                writer.add_scalar('Loss/Train_After', trn_after_perf[task, 0], task + 1)

                writer.add_scalar('Loss/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 0]), task + 1)
                writer.add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/CTM', CTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/LTM', LTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/Task_1', tst_after_perf[task, 0, 0], task + 1)

                ## Accuracy
                writer.add_scalar('Accuracy/Train_Before', trn_before_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/Train_After', trn_after_perf[task, 1], task + 1)

                writer.add_scalar('Accuracy/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 1]), task + 1)
                writer.add_scalar('Accuracy/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/CTM', CTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/LTM', LTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/Task_1', tst_after_perf[task, 0, 1], task + 1)

            online_mdls[task] = meta_model

            if save_models:
                torch.save(meta_model.state_dict(), '%s/Model/%s/%s/%d_meta_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
            np.save(path_ + '_tst_before_perf.npy', tst_before_perf)
            np.save(path_ + '_trn_before_perf.npy', trn_before_perf)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)


    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                meta_model = model_func().to(device)
                meta_model.load_state_dict(torch.load('%s/Model/%s/%s/%d_meta_model.pt'
                                                      % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = meta_model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
        tst_before_perf = np.load(path_ + '_tst_before_perf.npy')
        trn_before_perf = np.load(path_ + '_trn_before_perf.npy')

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_before_perf, trn_before_perf, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf


def train_MOML(data_obj, alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step,
               print_per, weight_decay, model_func, init_model, sch_step, sch_gamma, lr_decay_per_round,
               save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'MOML_' + suffix
    suffix += '_alpha%f_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f' % (
    alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks
    writer = SummaryWriter('%s/Runs/%s/%s' % (data_path, data_obj.name, suffix)) if save_tensorboard else None

    tst_before_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_before_perf = np.zeros((n_tasks, 2))

    tst_after_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((n_tasks, 2))

    CTM_perf = np.zeros((n_tasks, 2))
    LTM_perf = np.zeros((n_tasks, 2))

    # Initialize model
    meta_model = model_func().to(device)
    meta_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())), strict=False)

    omega_model = get_mdl_params([init_model])[0]
    n_par = omega_model.shape[0]
    lambda_model = torch.zeros(n_par, dtype=torch.float32, device=device)  # Start from all 0s

    if not os.path.exists('%s/Model/%s/%s/%d_tst_before_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Train
        for task in range(n_tasks):
            print('---- Round %2d' % (task + 1))

            ### Evaluation
            # Test all seen tasks including the current one before training.
            for tt in range(task):
                tst_before_perf[task][tt] = tst_after_perf[task - 1][
                    tt]  # The model is not updated, no need to calculate twice

            tst_before_perf[task][task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                            learning_rate_ft, num_grad_step, dataset_name,
                                                            tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            trn_before_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                      learning_rate_ft, num_grad_step, dataset_name)

            # Train only get the current task
            trn_x = task_x[task]
            trn_y = task_y[task]
            decay = lr_decay_per_round ** task
            meta_model = train_MOML_model(meta_model, model_func, trn_x, trn_y, alpha, omega_model, lambda_model,
                                          learning_rate * decay, learning_rate_ft, num_grad_step, batch_size, K,
                                          print_per,
                                          weight_decay, dataset_name, sch_step, sch_gamma)
            curr_par = get_mdl_params([meta_model], n_par=n_par)[0]
            # updating the lambda model and omega model
            lambda_model = lambda_model - alpha * (curr_par - omega_model)
            omega_model = 1 / 2 * (curr_par + omega_model) - 1 / 2 * 1 / alpha * lambda_model

            ### Evaluation
            trn_after_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                     learning_rate_ft, num_grad_step, dataset_name)

            # Test all seen tasks.
            for tt in range(task + 1):
                tst_after_perf[task][tt] = get_maml_acc_loss(task_x[tt], task_y[tt], meta_model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name,
                                                             tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])

            ### CTM and LTM
            CTM_perf[task] = tst_before_perf[task][task]
            LTM_perf[task] = np.mean(tst_after_perf[task, :task + 1, :], axis=0)

            print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                  % (task + 1, trn_after_perf[task][1], trn_after_perf[task][0]))
            print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                  % (task + 1, tst_after_perf[task, task, 1], tst_after_perf[task, task, 0]))

            if save_tensorboard:
                ## Loss
                writer.add_scalar('Loss/Train_Before', trn_before_perf[task, 0], task + 1)
                writer.add_scalar('Loss/Train_After', trn_after_perf[task, 0], task + 1)

                writer.add_scalar('Loss/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 0]), task + 1)
                writer.add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/CTM', CTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/LTM', LTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/Task_1', tst_after_perf[task, 0, 0], task + 1)

                ## Accuracy
                writer.add_scalar('Accuracy/Train_Before', trn_before_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/Train_After', trn_after_perf[task, 1], task + 1)

                writer.add_scalar('Accuracy/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 1]), task + 1)
                writer.add_scalar('Accuracy/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/CTM', CTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/LTM', LTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/Task_1', tst_after_perf[task, 0, 1], task + 1)

            online_mdls[task] = meta_model

            if save_models:
                torch.save(meta_model.state_dict(), '%s/Model/%s/%s/%d_meta_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
            np.save(path_ + '_tst_before_perf.npy', tst_before_perf)
            np.save(path_ + '_trn_before_perf.npy', trn_before_perf)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)


    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                meta_model = model_func().to(device)
                meta_model.load_state_dict(torch.load('%s/Model/%s/%s/%d_meta_model.pt'
                                                      % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = meta_model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
        tst_before_perf = np.load(path_ + '_tst_before_perf.npy')
        trn_before_perf = np.load(path_ + '_trn_before_perf.npy')

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_before_perf, trn_before_perf, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf

def train_BMOML(data_obj, alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step,
               print_per, weight_decay, model_func, init_model, sch_step, sch_gamma, lr_decay_per_round,
               save_models, save_performance, save_tensorboard, buffer_size=10, style='Last', suffix='', data_path=''):
    suffix = 'BMOML_' + style + suffix
    suffix += '_alpha%f_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f_bsize%d' % (
    alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay, buffer_size)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    buffer_list = []

    task_x = data_obj.trn_x
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks
    writer = SummaryWriter('%s/Runs/%s/%s' % (data_path, data_obj.name, suffix)) if save_tensorboard else None

    tst_before_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_before_perf = np.zeros((n_tasks, 2))

    tst_after_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((n_tasks, 2))

    CTM_perf = np.zeros((n_tasks, 2))
    LTM_perf = np.zeros((n_tasks, 2))

    # Initialize model
    meta_model = model_func().to(device)
    meta_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())), strict=False)

    omega_model = get_mdl_params([init_model])[0]
    n_par = omega_model.shape[0]
    lambda_model = torch.zeros(n_par, dtype=torch.float32, device=device)  # Start from all 0s

    if not os.path.exists('%s/Model/%s/%s/%d_tst_before_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Train
        for task in range(n_tasks):
            print('---- Round %2d' % (task + 1))

            ### Evaluation
            # Test all seen tasks including the current one before training.
            for tt in range(task):
                tst_before_perf[task][tt] = tst_after_perf[task - 1][
                    tt]  # The model is not updated, no need to calculate twice

            tst_before_perf[task][task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                            learning_rate_ft, num_grad_step, dataset_name,
                                                            tst_x=data_obj.tst_x[task], tst_y=data_obj.tst_y[task])

            trn_before_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                      learning_rate_ft, num_grad_step, dataset_name)

            #
            if style == 'Last':
                if task >= buffer_size:
                    trn_x = task_x[task-buffer_size+1:task+1]
                    trn_y = task_y[task-buffer_size+1:task+1]
                else:
                    trn_x = task_x[:task+1]
                    trn_y = task_y[:task+1]
            else:
                if task >= buffer_size:
                    cancel_idx = np.random.randint(buffer_size)
                    buffer_list[cancel_idx] = task
                    trn_x = task_x[buffer_list]
                    trn_y = task_y[buffer_list]
                else:
                    buffer_list.append(task)
                    trn_x = task_x[buffer_list]
                    trn_y = task_y[buffer_list]

            # trn_x = task_x[task]
            # trn_y = task_y[task]
            decay = lr_decay_per_round ** task

            meta_model = train_BMOML_model(meta_model, model_func, trn_x, trn_y, alpha, omega_model, lambda_model,
                                          learning_rate * decay, learning_rate_ft, num_grad_step, batch_size, K,
                                          print_per,
                                          weight_decay, dataset_name, sch_step, sch_gamma)
            curr_par = get_mdl_params([meta_model], n_par=n_par)[0]
            # updating the lambda model and omega model
            lambda_model = lambda_model - alpha * (curr_par - omega_model)
            omega_model = 1 / 2 * (curr_par + omega_model) - 1 / 2 * 1 / alpha * lambda_model

            ### Evaluation
            trn_after_perf[task] = get_maml_acc_loss(task_x[task], task_y[task], meta_model, model_func,
                                                     learning_rate_ft, num_grad_step, dataset_name)

            # Test all seen tasks.
            for tt in range(task + 1):
                tst_after_perf[task][tt] = get_maml_acc_loss(task_x[tt], task_y[tt], meta_model, model_func,
                                                             learning_rate_ft, num_grad_step, dataset_name,
                                                             tst_x=data_obj.tst_x[tt], tst_y=data_obj.tst_y[tt])

            ### CTM and LTM
            CTM_perf[task] = tst_before_perf[task][task]
            LTM_perf[task] = np.mean(tst_after_perf[task, :task + 1, :], axis=0)

            print('\n*** Task %2d, Training   data, Accuracy: %.4f, Loss: %.4f'
                  % (task + 1, trn_after_perf[task][1], trn_after_perf[task][0]))
            print('*** Task %2d, Test       data, Accuracy: %.4f, Loss: %.4f\n'
                  % (task + 1, tst_after_perf[task, task, 1], tst_after_perf[task, task, 0]))

            if save_tensorboard:
                ## Loss
                writer.add_scalar('Loss/Train_Before', trn_before_perf[task, 0], task + 1)
                writer.add_scalar('Loss/Train_After', trn_after_perf[task, 0], task + 1)

                writer.add_scalar('Loss/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 0]), task + 1)
                writer.add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/CTM', CTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/LTM', LTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/Task_1', tst_after_perf[task, 0, 0], task + 1)

                ## Accuracy
                writer.add_scalar('Accuracy/Train_Before', trn_before_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/Train_After', trn_after_perf[task, 1], task + 1)

                writer.add_scalar('Accuracy/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 1]), task + 1)
                writer.add_scalar('Accuracy/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/CTM', CTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/LTM', LTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/Task_1', tst_after_perf[task, 0, 1], task + 1)

            online_mdls[task] = meta_model

            if save_models:
                torch.save(meta_model.state_dict(), '%s/Model/%s/%s/%d_meta_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
            np.save(path_ + '_tst_before_perf.npy', tst_before_perf)
            np.save(path_ + '_trn_before_perf.npy', trn_before_perf)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)


    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                meta_model = model_func().to(device)
                meta_model.load_state_dict(torch.load('%s/Model/%s/%s/%d_meta_model.pt'
                                                      % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = meta_model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
        tst_before_perf = np.load(path_ + '_tst_before_perf.npy')
        trn_before_perf = np.load(path_ + '_trn_before_perf.npy')

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_before_perf, trn_before_perf, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf


