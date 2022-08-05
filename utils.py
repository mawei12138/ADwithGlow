import os
import torch
from torchvision import datasets, transforms
import config as c
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import CIFAR10, STL10


# from lbot_dataset import *
# from thop import profile
# from thop import clever_format
def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


# def flat(tensor):
#     return tensor.reshape(tensor.shape[0], -1)


# def concat_maps(maps):
#     flat_maps = list()
#     for m in maps:
#         flat_maps.append(flat(m))
#     return torch.cat(flat_maps, dim=1)[..., None]


# def get_loss(z, jac):
#     z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
#     jac = sum(jac)
#     return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]
def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    z = z.reshape(z.shape[0],-1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


def load_datasets(dataset_path, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_path>:

    train data:

            dataset_path/class_name/train/good/any_filename.png
            dataset_path/class_name/train/good/another_filename.tif
            dataset_path/class_name/train/good/xyz.png
            [...]

    test data:

        'normal data' = non-anomalies

            dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
            dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
            dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
            dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
            dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
            [...]

        anomalies - assume there are anomaly classes 'crack' and 'curved'

            dataset_path/class_name/test/crack/dat_crack_damn.png
            dataset_path/class_name/test/crack/let_it_crack.png
            dataset_path/class_name/test/crack/writing_docs_is_fun.png
            [...]

            dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
            dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
            [...]
    '''

    def target_transform(target):
        return class_perm[target]
    if c.dataset == 'mvtec':
        data_dir_train = os.path.join(dataset_path, class_name, 'train')
        data_dir_test = os.path.join(dataset_path, class_name, 'test')

        classes = os.listdir(data_dir_test)
        if 'good' not in classes:
            print(
                'There should exist a subdirectory "good". Read the doc of this function for further information.')
            exit()
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if cl == 'good':
                class_perm.append(0)
            else:
                class_perm.append(class_idx)
                class_idx += 1

        tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]
        transform_train = transforms.Compose(tfs)
        trainset = ImageFolder(data_dir_train, transform=transform_train)
        testset = ImageFolder(data_dir_test, transform=transform_train, target_transform=target_transform)
    elif c.dataset == 'cifar10':
        c.img_size = (768, 768)
        tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]
        transform_train = transforms.Compose(tfs)
        train_set = CIFAR10(root='./data1/cifar10', train=True, download=True, transform=transform_train)
        test_set = CIFAR10(root='./data1/cifar10', train=False, download=True, transform=transform_train)
        n_idx = train_set.class_to_idx[class_name]
        # train_set, test_set = get_cifar_anomaly_dataset(train_set, test_set, n_idx)
        train_set, test_set = get_cifar_small_anomaly_dataset(train_set, test_set, n_idx)
        return train_set, test_set
    else:
        raise AttributeError
    return trainset, testset


def make_dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels

    # def get_model_params(model,input):
    #     flops,params = (model,input)
    #     flops,params = clever_format(flops,params)
    #     print(flops,params)
    #     total = sum([param.nelement() for param in model.parameters()])
    #     print("Number of paramater : %.2fM" % (total/1e6))
    # class Score_Observer:
    #     '''Keeps an eye on the current and highest score so far'''
    #
    #     def __init__(self, name):
    #         self.name = name
    #         self.max_epoch = 0
    #         self.max_score = None
    #         self.min_loss_epoch = 0
    #         self.min_loss_score = 0
    #         self.min_loss = None
    #         self.last = None
    #
    #     def update(self, score, epoch, print_score=False):
    #         self.last = score
    #         if epoch == 0 or score > self.max_score:
    #             self.max_score = score
    #             self.max_epoch = epoch
    #         if print_score:
    #             self.print_score()
    #
    #     def print_score(self):
    #         print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d} \t epoch_loss: {:d}'.format(self.name, self.last,
    # self.max_score,
    # self.max_epoch,
    # self.min_loss_epoch))


def get_cifar_anomaly_dataset(train_ds, valid_ds, n_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - CIFAR10} -- Training dataset
        valid_ds {Dataset - CIFAR10} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl == n_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl != n_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]  # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl == n_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != n_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . ->         . -> nortest
    #     #mal
    #        . -> abnormal
    # train_ds.data = np.copy(nrm_trn_img)[:100]
    # valid_ds.data = np.concatenate((nrm_tst_img[:50], abn_trn_img[:25], abn_tst_img[:25]), axis=0)
    # train_ds.targets = np.copy(nrm_trn_lbl)[:100]
    # valid_ds.targets = np.concatenate((nrm_tst_lbl[:50], abn_trn_lbl[:25], abn_tst_lbl[:25]), axis=0)
    train_ds.data = np.copy(nrm_trn_img)
    # valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)
    train_ds.targets = np.copy(nrm_trn_lbl)
    # valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
    return train_ds, valid_ds


def get_STL10_anomaly_dataset(train_ds, valid_ds, n_cls_idx=0):
    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.labels)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.labels)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl == n_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl != n_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels

    nrm_tst_idx = np.where(tst_lbl == n_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != n_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . ->         . -> nortest
    #     #mal
    #        . -> abnormal
    # train_ds.data = np.copy(nrm_trn_img)[:100]
    # valid_ds.data = np.concatenate((nrm_tst_img[:50], abn_trn_img[:25], abn_tst_img[:25]), axis=0)
    # train_ds.targets = np.copy(nrm_trn_lbl)[:100]
    # valid_ds.targets = np.concatenate((nrm_tst_lbl[:50], abn_trn_lbl[:25], abn_tst_lbl[:25]), axis=0)
    train_ds.data = np.copy(nrm_trn_img)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)
    train_ds.lables = np.copy(nrm_trn_lbl)
    valid_ds.labels = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)

    return train_ds, valid_ds


def get_cifar_small_anomaly_dataset(train_ds, valid_ds, n_cls_idx=0):
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    nrm_trn_idx = np.where(trn_lbl == n_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl != n_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels

    nrm_tst_idx = np.where(tst_lbl == n_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != n_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.
    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_tst_lbl[:] = 1
    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . ->         . -> nortest
    #     #mal
    #        . -> abnormal
    train_ds.data = np.copy(nrm_trn_img)[:50]
    valid_ds.data = np.concatenate((nrm_tst_img[:10], abn_tst_img[:90]), axis=0)
    train_ds.targets = np.copy(nrm_trn_lbl)[:50]
    valid_ds.targets = np.concatenate((nrm_tst_lbl[:10], abn_tst_lbl[:90]), axis=0)

    return train_ds, valid_ds
