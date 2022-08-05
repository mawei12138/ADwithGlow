'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cpu'  # or 'cpu'

# data settings
dataset_path = "./data/"  # parent directory of datasets
dataset = 'mvtec' # 'cifar10'|'STL10'
class_name = "dummy_class"  # dataset subdirectory
modelname = "dummy_test"  # export evaluations/logs with this name
subnet = None
# img_size = (768, 768)  # image size of highest scale, others are //2, //4
img_size = (256,256)
# img_dims = [3] + list(img_size)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
clamp = 3  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 1  # higher = more flexible = more unstable
fc_internal = 512  # * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 2e-4  # inital learning rate
use_gamma = True

extractor = "wide_resnet"  # feature dataset name (which was used in 'extract_features.py' as 'export_name')
n_feat = {"effnetB5": 512,'wide_resnet':1024}[extractor]  # dependend from feature extractor
# map_size = (img_size[0] // 12, img_size[1] // 12)

# dataloader parameters
batch_size = 16  # actual batch size is this value multiplied by n_transforms(_test)
# kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
kernel_sizes = [1] * (n_coupling_blocks - 1) + [3]

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 1  # total epochs = meta_epochs * sub_epochs
sub_epochs = 1  # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True
