from utils_libs import *
from utils_dataset import *
from utils_models import *

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
import secrets

import time

max_norm = 3
momentum = 0.0

# --- Helper Methods
def get_mdl_params(model_list, n_par=0):
    if n_par == 0:
        for name, param in model_list[0].named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = torch.zeros((len(model_list), n_par), dtype=torch.float32, device=device)

    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.detach().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)

    return param_mat

def get_mdl_grads(model_list, n_par=0):
    if n_par == 0:
        for name, param in model_list[0].named_parameters():
            n_par += len(param.data.reshape(-1))

    grad_mat = torch.zeros((len(model_list), n_par), dtype=torch.float32, device=device)

    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.grad.detach().reshape(-1)
            grad_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)

    return grad_mat


def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_load = torch.utils.data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        for data in tst_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            y_pred = model(batch_x)
            # Loss calculation
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model]).cpu().numpy()
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


def get_maml_acc_loss(trn_x, trn_y, model, model_func, learning_rate, num_grad_step, dataset_name, tst_x=False,
                      tst_y=False, weight_decay=0, weight_decay_tst=False, batch_sz=None):
    _model = model_func().to(device)
    _model.load_state_dict(copy.deepcopy(dict(model.named_parameters())),strict=False)
    for params in _model.parameters():
        params.requires_grad = True

    optimizer_ = torch.optim.SGD(_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Do Fine Tuning on all dataset
    if batch_sz is None:
        batch_sz = len(trn_y)
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_sz, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    _model.train();
    _model = _model.to(device)

    for _ in range(num_grad_step):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            y_pred = _model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            optimizer_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=_model.parameters(), max_norm=max_norm)  # Clip gradients
            optimizer_.step()

    weight_decay_tst = weight_decay if isinstance(weight_decay_tst, bool) else weight_decay_tst
    # Get train acc
    if isinstance(tst_x, bool):
        loss_, acc_ = get_acc_loss(trn_x, trn_y, _model, dataset_name, weight_decay)
    else:
        loss_, acc_ = get_acc_loss(tst_x, tst_y, _model, dataset_name, weight_decay_tst)

    # Free memory
    del _model
    return loss_, acc_

# --- Training methods
def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name,
                sch_step, sch_gamma):
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train();
    model = model.to(device)

    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)

    print_test = not isinstance(tst_x, bool)  # Put tst_x=False if no tst data given
    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    print_ = "Step %4d, LR: %.4f, Training Acc: %.4f, Loss: %.4f" % (0, scheduler_.get_lr()[0], acc_trn, loss_trn)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print_ += ", Test Acc: %.4f, Loss: %.4f" % (acc_tst, loss_tst)
    print(print_)

    k = 0;
    epoch = 0
    while (k < K):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]

            optimizer_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients

            optimizer_.step()

            k += 1

            if k % print_per == 0:
                loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
                print_ = "Step %4d, LR: %.4f, Training Acc: %.4f, Loss: %.4f" % (
                k, scheduler_.get_lr()[0], acc_trn, loss_trn)
                if print_test:
                    loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                    print_ += ", Test Acc: %.4f, Loss: %.4f" % (acc_tst, loss_tst)
                print(print_)

            if k == K:
                break

        scheduler_.step()

        epoch += 1;
        print('- Epoch %3d' % epoch)

    return model

###
def train_FTML_model(model, model_func, trn_x, trn_y, learning_rate, learning_rate_ft, num_grad_step, batch_size, K,
                     print_per, weight_decay, dataset_name, sch_step, sch_gamma):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train();
    model = model.to(device)

    #optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)
    inner_opt = torch.optim.SGD(model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)
    t_task = len(trn_x)

    for k in range(K):
        # Sample one task at uniformly random
        cur_task = np.random.randint(t_task)
        cur_task_x = trn_x[cur_task]
        cur_task_y = trn_y[cur_task]
        # Get train and validation sets
        trn_load = torch.utils.data.DataLoader(
            Dataset(cur_task_x, cur_task_y, train=True, dataset_name=dataset_name),
            batch_size=batch_size, shuffle=True)

        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(curr_trn_x)
                trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            val_logits = fnet(curr_val_x)
            val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            val_loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
        optimizer_.step()

        if (k + 1) % print_per == 0:
            loss_trn, acc_trn = get_maml_acc_loss(cur_task_x, cur_task_y, model, model_func, learning_rate_ft,
                                                  num_grad_step, dataset_name, weight_decay=weight_decay)
            lr_print = 'Chosen task %3d, LR, meta: %.4f' % (cur_task + 1, scheduler_.get_lr()[0])
            print_ = "Step %4d, %s, Training Acc: %.4f, Loss: %.4f" % ((k + 1), lr_print, acc_trn, loss_trn)
            print(print_)

        scheduler_.step()

    return model

def train_MOGD_model(model, model_func, trn_x, trn_y, learning_rate, learning_rate_ft, num_grad_step, batch_size, K,
                     print_per, weight_decay, dataset_name, sch_step, sch_gamma):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train();
    model = model.to(device)

    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)
    inner_opt = torch.optim.SGD(model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True)

    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(curr_trn_x)
                trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            val_logits = fnet(curr_val_x)
            val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            val_loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
        optimizer_.step()

        if (k + 1) % print_per == 0:
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, learning_rate_ft,
                                                  num_grad_step, dataset_name, weight_decay=weight_decay)
            lr_print = 'LR, meta: %.4f' % scheduler_.get_lr()[0]
            print_ = "Step %4d, %s, Training Acc: %.4f, Loss: %.4f" % ((k + 1), lr_print, acc_trn, loss_trn)
            print(print_)

        scheduler_.step()

    return model

def train_MOML_model(model, model_func, trn_x, trn_y, alpha, omega_model, lambda_model, learning_rate,
                             learning_rate_ft, num_grad_step, batch_size, K, print_per, weight_decay, dataset_name,
                              sch_step, sch_gamma):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train()
    model = model.to(device)

    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)
    inner_opt = torch.optim.SGD(model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)
    # Get train and validation sets
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True)
    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        # sample a batch of support data and a batch of query data
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(curr_trn_x)
                trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            val_logits = fnet(curr_val_x)
            val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)
            loss_lambda = -torch.sum(mld_pars * lambda_model)
            loss_server = -alpha * torch.sum(mld_pars * omega_model) + alpha / 2 * torch.sum(mld_pars * mld_pars)
            val_loss = val_loss + loss_lambda + loss_server

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            val_loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
        optimizer_.step()

        if (k + 1) % print_per == 0:
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, learning_rate_ft,
                                                  num_grad_step, dataset_name, weight_decay=weight_decay)
            lr_print = 'LR, meta: %.4f' % scheduler_.get_lr()[0]
            print_ = "Step %4d, %s, Training Acc: %.4f, Loss: %.4f" % ((k + 1), lr_print, acc_trn, loss_trn)
            print(print_)

        scheduler_.step()

    return model


def train_MOML_model_variant1(model, model_func, trn_x, trn_y, gamma_model, delta_model, learning_rate, rho,
                              learning_rate_ft, num_grad_step, batch_size, K, print_per, weight_decay, dataset_name,
                              sch_step=1, sch_gamma=1):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model = copy.deepcopy(gamma_model)
    model.train()
    model = model.to(device)
    gamma_model = gamma_model.to(device)

    # Get derivative of the adapted model.
    trn_load_labmda = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                                  batch_size=trn_x.shape[0], shuffle=True)

    lambda_model = torch.zeros(delta_model.shape[0], dtype=torch.float32, device=device)
    optimizer_l = torch.optim.SGD(gamma_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    inner_opt_l = torch.optim.SGD(gamma_model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)

    for data in trn_load_labmda:
        all_x, all_y = data[0].to(device), data[1].to(device)
        # Higher library
        optimizer_l.zero_grad()
        with higher.innerloop_ctx(gamma_model, inner_opt_l, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(all_x)
                trn_loss = loss_fn(trn_logits, all_y.reshape(-1).long()) / list(all_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            all_logits = fnet(all_x)
            all_loss = loss_fn(all_logits, all_y.reshape(-1).long())

            all_loss.backward()
            idx = 0
            for name, param in gamma_model.named_parameters():
                temp = param.grad.detach().reshape(-1)
                lambda_model[idx:idx + len(temp)] += temp / trn_x.shape[0]
                idx += len(temp)

    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)
    inner_opt = torch.optim.SGD(model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)

    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True)

    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        # sample a batch of support data and a batch of query data
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(curr_trn_x)
                trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            val_logits = fnet(curr_val_x)
            val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)

            loss_lambda = (rho - 1) * torch.sum(mld_pars * lambda_model)
            loss_delta = (1 - rho) * torch.sum(mld_pars * delta_model)
            val_loss = val_loss + loss_lambda + loss_delta

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            val_loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
        optimizer_.step()

        if (k + 1) % print_per == 0:
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, learning_rate_ft,
                                                  num_grad_step, dataset_name, weight_decay=weight_decay)
            lr_print = 'LR, meta: %.4f' % (scheduler_.get_lr()[0])
            print_ = "Step %4d, %s, Training Acc: %.4f, Loss: %.4f" % ((k + 1), lr_print, acc_trn, loss_trn)
            print(print_)

        scheduler_.step()

    # update the delta model
    delta_model = rho * lambda_model + (1 - rho) * delta_model
    return model, delta_model

def train_BMOML_model(model, model_func, trn_x, trn_y, alpha, omega_model, lambda_model, learning_rate,
                             learning_rate_ft, num_grad_step, batch_size, K, print_per, weight_decay, dataset_name,
                              sch_step, sch_gamma):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train()
    model = model.to(device)

    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=sch_step, gamma=sch_gamma)
    inner_opt = torch.optim.SGD(model.parameters(), lr=learning_rate_ft, weight_decay=weight_decay)
    # Get train and validation sets
    buffer_size = len(trn_x)

    for k in range(K):
        cur_task = np.random.randint(buffer_size)
        cur_task_x = trn_x[cur_task]
        cur_task_y = trn_y[cur_task]
        # Get train and validation sets
        trn_load = torch.utils.data.DataLoader(
            Dataset(cur_task_x, cur_task_y, train=True, dataset_name=dataset_name),
            batch_size=batch_size, shuffle=True)


        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        # sample a batch of support data and a batch of query data
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(num_grad_step):
                trn_logits = fnet(curr_trn_x)
                trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                diffopt.step(trn_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            val_logits = fnet(curr_val_x)
            val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)
            loss_lambda = -torch.sum(mld_pars * lambda_model)
            loss_server = -alpha * torch.sum(mld_pars * omega_model) + alpha / 2 * torch.sum(mld_pars * mld_pars)
            val_loss = val_loss + loss_lambda + loss_server

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            val_loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
        optimizer_.step()

        if (k + 1) % print_per == 0:
            loss_trn, acc_trn = get_maml_acc_loss(cur_task_x, cur_task_y, model, model_func, learning_rate_ft,
                                                  num_grad_step, dataset_name, weight_decay=weight_decay)
            lr_print = 'LR, meta: %.4f' % scheduler_.get_lr()[0]
            print_ = "Step %4d, %s, Training Acc: %.4f, Loss: %.4f" % ((k + 1), lr_print, acc_trn, loss_trn)
            print(print_)

        scheduler_.step()

    return model