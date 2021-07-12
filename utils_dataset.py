from utils_libs import *

dataset_path = "/home/rzhu/datasets/"


class DatasetObject:
    def __init__(self, dataset, seed, n_task, rule, rule_arg='', data_path=''):
        self.dataset = dataset
        self.rule = rule
        self.seed = seed
        self.n_task = n_task
        self.name = "%s_%d_%s_%d" % (self.dataset, self.seed, self.rule, self.n_task)
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        rule_arg_str = rule_arg_str if rule_arg == '' else '_' + rule_arg_str
        self.rule_arg = rule_arg
        self.name = self.name + rule_arg_str
        self.data_path = data_path
        self.set_data()

    # Task generation methods
    def rotation90(self, cur_img, n_rotations):
        return np.rot90(cur_img, k=n_rotations, axes=(1, 2))  ## Apply rotations

    def flip_hor_ver(self, cur_img, flip):
        cur_img = np.flip(cur_img, axis=2) if flip == -1 else cur_img  ## Horizontal flip
        cur_img = np.flip(cur_img, axis=1) if flip == +1 else cur_img  ## Vertical   flip
        return cur_img

    def scale_img(self, cur_img, scale, width_=26, ratio=2):
        if scale == 1:
            img_width = cur_img.shape[1]
            rem = (img_width - width_) // ratio

            # Get middle width_Xwidth_ and
            mid_ = cur_img[0, rem:-rem, rem:-rem]
            # Get the mean value for the other parts
            val_ = (np.sum(cur_img) - np.sum(mid_)) / (img_width * img_width - width_ * width_)
            # Rescale mid
            h_width = width_ // ratio
            q_width = width_ // (ratio * 2)

            mid_ = np.array(Image.fromarray(mid_).resize((h_width, h_width)))

            cur_img[0, rem:-rem, rem:-rem] = val_
            cur_img[0, rem + q_width:rem + q_width + h_width, rem + q_width:rem + q_width + h_width] = mid_
        return cur_img

    def crop_img(self, cur_img, crop, pad=4):
        # Right or bottom padding
        if crop == -2:
            cur_img = cur_img[:, ::-1, :]
        if crop == +2:
            cur_img = cur_img[:, :, ::-1]

        height = self.height + (crop < 0) * pad
        width = self.width + (crop > 0) * pad
        pad_h = pad * (crop < 0)
        pad_w = pad * (crop > 0)

        extended_img = np.zeros((self.channels, height, width)).astype(np.float32)
        extended_img[:, pad_h:, pad_w:] = cur_img
        cur_img = extended_img[:, :self.height, :self.width]

        if crop == -2:
            cur_img = cur_img[:, ::-1, :]
        if crop == +2:
            cur_img = cur_img[:, :, ::-1]

        return cur_img

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%s/Data/%s' % (self.data_path, self.name)):
            # Get Raw data
            # torchvision dataset
            if self.dataset in ['mnist', 'CIFAR100']:
                if self.dataset == 'mnist':
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                    trnset = torchvision.datasets.MNIST(root=dataset_path,
                                                        train=True, download=True, transform=transform)
                    tstset = torchvision.datasets.MNIST(root=dataset_path,
                                                        train=False, download=True, transform=transform)

                    trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                    tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                    self.channels = 1;
                    self.width = 28;
                    self.height = 28;
                    self.n_cls = 10;

                if self.dataset == 'CIFAR100':
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                         std=[0.2675, 0.2565, 0.2761])])
                    trnset = torchvision.datasets.CIFAR100(root=dataset_path,
                                                           train=True, download=True, transform=transform)
                    tstset = torchvision.datasets.CIFAR100(root=dataset_path,
                                                           train=False, download=True, transform=transform)
                    trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                    tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                    self.channels = 3;
                    self.width = 32;
                    self.height = 32;
                    self.n_cls = 100;

                trn_itr = trn_load.__iter__();
                tst_itr = tst_load.__iter__()
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy();
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy();
                tst_y = tst_y.numpy().reshape(-1, 1)

                # Shuffle Data
                np.random.seed(self.seed)
                rand_perm = np.random.permutation(len(trn_y))
                trn_x = trn_x[rand_perm]
                trn_y = trn_y[rand_perm]

                rand_perm = np.random.permutation(len(tst_y))
                tst_x = tst_x[rand_perm]
                tst_y = tst_y[rand_perm]

            else:
                # Manually added dataset
                if self.dataset == 'miniImageNet':
                    # There are train.train, train.val, train.test, val and test files.
                    # Train.train, val and test files has different classes with 600 datapoints each.
                    # Combine train.train, val and test files into 60K images.
                    # Split this as train and test as in CIFAR100 dataset. Train 50K, Test 50K
                    total_data = []
                    path_list = []
                    path_list.append(dataset_path + "miniimagenet_numpy/miniImageNet_category_split_train_phase_train.pickle")
                    path_list.append(dataset_path + "miniimagenet_numpy/miniImageNet_category_split_val.pickle")
                    path_list.append(dataset_path + "miniimagenet_numpy/miniImageNet_category_split_test.pickle")

                    for file in path_list:
                        try:
                            with open(file, 'rb') as fo:
                                data = pickle.load(fo)
                        except:
                            with open(file, 'rb') as f:
                                u = pickle._Unpickler(f)
                                u.encoding = 'latin1'
                                data = u.load()
                        total_data.append(data)

                    trn_trn = total_data[0]
                    val_ = total_data[1]
                    tst_ = total_data[2]
                    data_x = np.concatenate((trn_trn['data'], val_['data'], tst_['data']), axis=0)
                    data_y = np.concatenate(
                        (np.asarray(trn_trn['labels']), np.asarray(val_['labels']), np.asarray(tst_['labels'])),
                        axis=0).reshape(-1, 1)

                    # Get idx of classes
                    n_cls = 100
                    cls_idx_list = list(range(n_cls))
                    for i in range(n_cls):
                        cls_idx_list[i] = np.where(data_y[:, 0] == i)[0]

                    np.random.seed(self.seed)
                    trn_idx = [];
                    tst_idx = []
                    trn_per_cls = 500
                    for i in range(n_cls):
                        curr_list = cls_idx_list[i]
                        np.random.shuffle(curr_list)
                        trn_idx.extend(curr_list[:trn_per_cls])
                        tst_idx.extend(curr_list[trn_per_cls:])

                    # Set trn and tst, make images as Channel Height Width style
                    trn_x = np.moveaxis(data_x[trn_idx], source=3, destination=1)
                    trn_y = data_y[trn_idx]

                    tst_x = np.moveaxis(data_x[tst_idx], source=3, destination=1)
                    tst_y = data_y[tst_idx]

                    mean_ = np.mean(trn_x, axis=(0, 2, 3))
                    std_ = np.std(trn_x, axis=(0, 2, 3))

                    # Keep it in range 0-255 for the data augmentation part
                    # PIL image takes it as 8 bytes (0-255 pixels) so normalize at the end.
                    # Divide these numbers with 255 since, we will normalize after transforming to Tensor.
                    DatasetObject.miniImageNet_mean = mean_ / 255
                    DatasetObject.miniImageNet_std = std_ / 255

                    self.channels = 3;
                    self.width = 84;
                    self.height = 84;
                    self.n_cls = 100;

                    np.random.seed(self.seed)

            if self.rule == 'mnist_rotations_flips_crop_scale':
                n_trn_data_per_task = len(trn_x) // self.n_task;
                excess_trn = len(trn_x) - n_trn_data_per_task * self.n_task
                n_tst_data_per_task = len(tst_x) // self.n_task;
                excess_tst = len(tst_x) - n_tst_data_per_task * self.n_task

                # Initialize trn and tst arrays
                self.trn_x = [np.zeros((n_trn_data_per_task + (self.n_task - task_ <= excess_trn), self.channels,
                                        self.height, self.width)).astype(np.float32) for task_ in range(self.n_task)]
                self.trn_y = [np.zeros((n_trn_data_per_task + (self.n_task - task_ <= excess_trn), 1)).astype(np.int64)
                              for task_ in range(self.n_task)]

                self.tst_x = [np.zeros((n_tst_data_per_task + (self.n_task - task_ <= excess_tst), self.channels,
                                        self.height, self.width)).astype(np.float32) for task_ in range(self.n_task)]
                self.tst_y = [np.zeros((n_tst_data_per_task + (self.n_task - task_ <= excess_tst), 1)).astype(np.int64)
                              for task_ in range(self.n_task)]

                # Do rotations, flips, crops, scale randomly for each task
                # Generate task generation rules for each class
                task_rotations = np.random.randint(4, size=(self.n_task, self.n_cls))  # 0,1,2,3
                task_flips = np.random.randint(3, size=(self.n_task, self.n_cls)) - 1  # -1,0,1
                task_scales = np.random.randint(2, size=(self.n_task, self.n_cls))  # 0,1
                task_crop = np.random.randint(5, size=(self.n_task, self.n_cls)) - 2  # -2,-1,0,1,2

                n_trn_all = 0;
                n_tst_all = 0
                for task in range(self.n_task):
                    # Generate train tasks
                    n_trn = len(self.trn_x[task])
                    self.trn_x[task] = trn_x[n_trn_all: n_trn_all + n_trn]
                    self.trn_y[task] = trn_y[n_trn_all: n_trn_all + n_trn]
                    n_trn_all += n_trn
                    for idx in range(n_trn):
                        cur_img = self.trn_x[task][idx]
                        cur_lbl = self.trn_y[task][idx][0]
                        cur_img = self.rotation90(cur_img, task_rotations[task][cur_lbl])
                        cur_img = self.flip_hor_ver(cur_img, task_flips[task][cur_lbl])
                        cur_img = self.scale_img(cur_img, task_scales[task][cur_lbl], width_=26, ratio=2)
                        cur_img = self.crop_img(cur_img, task_crop[task][cur_lbl], pad=4)
                        self.trn_x[task][idx] = cur_img

                    # Generate test  tasks
                    n_tst = len(self.tst_x[task])
                    self.tst_x[task] = tst_x[n_tst_all: n_tst_all + n_tst]
                    self.tst_y[task] = tst_y[n_tst_all: n_tst_all + n_tst]
                    n_tst_all += n_tst
                    for idx in range(n_tst):
                        cur_img = self.tst_x[task][idx]
                        cur_lbl = self.tst_y[task][idx][0]
                        cur_img = self.rotation90(cur_img, task_rotations[task][cur_lbl])
                        cur_img = self.flip_hor_ver(cur_img, task_flips[task][cur_lbl])
                        cur_img = self.scale_img(cur_img, task_scales[task][cur_lbl], width_=26, ratio=2)
                        cur_img = self.crop_img(cur_img, task_crop[task][cur_lbl], pad=4)
                        self.tst_x[task][idx] = cur_img
            elif self.rule == 'way':
                n_way = self.rule_arg
                assert self.n_task % self.n_cls == 0, 'Error, How to distribute the classes?'
                cls_occurance = self.n_task // self.n_cls * n_way

                cls_occ_list = (np.ones(self.n_cls) * cls_occurance).astype(np.int32)
                cls_list = []
                for task in range(self.n_task):
                    curr_list = []
                    while len(curr_list) != n_way:
                        # Get classes that is not chosen as of now
                        max_val = np.max(cls_occ_list)
                        idx_ = np.where(cls_occ_list == max_val)[0]
                        _list = np.arange(len(idx_))
                        np.random.shuffle(_list)
                        min_size = min(len(_list), n_way - len(curr_list))
                        for ii in range(min_size):
                            curr_list.append(idx_[_list[ii]])
                            cls_occ_list[idx_[_list[ii]]] -= 1
                    # print(curr_list)
                    cls_list.append(curr_list)
                self.cls_list = np.asarray(cls_list)
                self.trn_x = [];
                self.trn_y = []
                self.tst_x = [];
                self.tst_y = []

                trn_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount_trn = [len(trn_list[i]) for i in range(self.n_cls)]
                cls_per_task_trn = [len(trn_list[i]) // cls_occurance for i in range(self.n_cls)]

                tst_list = [np.where(tst_y == i)[0] for i in range(self.n_cls)]
                cls_amount_tst = [len(tst_list[i]) for i in range(self.n_cls)]
                cls_per_task_tst = [len(tst_list[i]) // cls_occurance for i in range(self.n_cls)]

                for cls in range(self.n_cls):
                    assert cls_per_task_trn[cls] * cls_occurance == cls_amount_trn[cls], 'Remaining points in train..'
                    assert cls_per_task_tst[cls] * cls_occurance == cls_amount_tst[cls], 'Remaining points in test...'

                cls_occ_list = np.zeros(self.n_cls).astype(np.int32)

                for task in range(self.n_task):
                    cur_trn_x = [];
                    cur_trn_y = []
                    cur_tst_x = [];
                    cur_tst_y = []
                    for jj, cls in enumerate(cls_list[task]):
                        cur_trn_x.extend(
                            np.asarray(trn_x[trn_list[cls]][
                                       cls_per_task_trn[cls] * cls_occ_list[cls]:cls_per_task_trn[cls] * (
                                                   1 + cls_occ_list[cls])]).astype(np.float32)
                        )
                        cur_trn_y.extend(
                            np.ones((cls_per_task_trn[cls], 1)).astype(np.float32) * jj
                        )

                        cur_tst_x.extend(
                            np.asarray(tst_x[tst_list[cls]][
                                       cls_per_task_tst[cls] * cls_occ_list[cls]:cls_per_task_tst[cls] * (
                                                   1 + cls_occ_list[cls])]).astype(np.float32)
                        )
                        cur_tst_y.extend(
                            np.ones((cls_per_task_tst[cls], 1)).astype(np.float32) * jj
                        )

                        cls_occ_list[cls] += 1

                    self.trn_x.append(cur_trn_x);
                    self.trn_y.append(cur_trn_y)
                    self.tst_x.append(cur_tst_x);
                    self.tst_y.append(cur_tst_y)

                self.trn_x = np.asarray(self.trn_x).astype(np.float32)
                self.trn_y = np.asarray(self.trn_y).astype(np.float32)

                self.tst_x = np.asarray(self.tst_x).astype(np.float32)
                self.tst_y = np.asarray(self.tst_y).astype(np.float32)
            else:
                assert False, 'Error in rule %s' % self.rule

            # Save data
            os.mkdir('%s/Data/%s' % (self.data_path, self.name))

            np.save('%s/Data/%s/trn_x.npy' % (self.data_path, self.name), self.trn_x)
            np.save('%s/Data/%s/trn_y.npy' % (self.data_path, self.name), self.trn_y)

            np.save('%s/Data/%s/tst_x.npy' % (self.data_path, self.name), self.tst_x)
            np.save('%s/Data/%s/tst_y.npy' % (self.data_path, self.name), self.tst_y)

            if self.dataset == 'CIFAR100' or self.dataset == 'miniImageNet':
                # Actual Label information
                np.save('%s/Data/%s/cls_list.npy' % (self.data_path, self.name), self.cls_list)
            if self.dataset == 'miniImageNet':
                np.save('%s/Data/%s/mean_.npy' % (self.data_path, self.name), DatasetObject.miniImageNet_mean)
                np.save('%s/Data/%s/std_.npy' % (self.data_path, self.name), DatasetObject.miniImageNet_std)

        else:
            print("Data is already downloaded")
            self.trn_x = np.load('%s/Data/%s/trn_x.npy' % (self.data_path, self.name))
            self.trn_y = np.load('%s/Data/%s/trn_y.npy' % (self.data_path, self.name))

            self.n_task = len(self.trn_x)

            self.tst_x = np.load('%s/Data/%s/tst_x.npy' % (self.data_path, self.name))
            self.tst_y = np.load('%s/Data/%s/tst_y.npy' % (self.data_path, self.name))

            if self.dataset == 'CIFAR100' or self.dataset == 'miniImageNet':
                self.cls_list = np.load('%s/Data/%s/cls_list.npy' % (self.data_path, self.name))

            if self.dataset == 'miniImageNet':
                DatasetObject.miniImageNet_mean = np.load('%s/Data/%s/mean_.npy' % (self.data_path, self.name))
                DatasetObject.miniImageNet_std = np.load('%s/Data/%s/std_.npy' % (self.data_path, self.name))

            if self.dataset == 'mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;
            if self.dataset == 'miniImageNet':
                self.channels = 3;
                self.width = 84;
                self.height = 84;
                self.n_cls = 100;

        print('Class frequencies:')
        count = 0
        print('-------- Train')
        for task in range(self.n_task):
            print("Task %3d: " % (task + 1) +
                  ', '.join(["%.3f" % np.mean(self.trn_y[task] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.trn_y[task].shape[0])
            count += self.trn_y[task].shape[0]

        print('Total Amount:%d' % count)
        print('-------- Test')
        count = 0
        for task in range(self.n_task):
            print("Task %3d: " % (task + 1) +
                  ', '.join(["%.3f" % np.mean(self.tst_y[task] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.tst_y[task].shape[0])
            count += self.tst_y[task].shape[0]
        print('Total Amount:%d' % count)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name == 'miniImageNet':
            self.X_data = data_x.astype(np.uint8)  # In range 0-255
            self.X_data = np.moveaxis(self.X_data, source=1, destination=3)  # Make it H,W,C
            self.y_data = data_y

            self.train = train
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DatasetObject.miniImageNet_mean, std=DatasetObject.miniImageNet_std)
            ])
            self.noaugmt_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DatasetObject.miniImageNet_mean, std=DatasetObject.miniImageNet_std)
            ])

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        assert isinstance(idx,
                          int), 'Expecting an index here. Some of the below part might not function correctly on a batch...'
        if self.name == 'mnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if (np.random.rand() > .5):
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'miniImageNet':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)

            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
