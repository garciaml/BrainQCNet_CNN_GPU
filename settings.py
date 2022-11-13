base_architecture = 'resnet152'
img_size = 224
num_classes = 2
add_on_layers_type = 'regular'

experiment_run = 'resnet152_lr_addonlayer_1e-4_11-10-2022'

data_path = '../QA_project/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'validation_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 3
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 1e-4}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 1e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 50
num_warm_epochs = 2

push_start = 5
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]
