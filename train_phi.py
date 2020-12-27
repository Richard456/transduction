import random
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append("../")
from data_utils import ADVMNISTLoader
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# from test import test

source_dataset_name = 'MNIST'
target_dataset_name = 'adv_mnist'
data_root = "../dataset"
source_image_root = os.path.join(data_root, source_dataset_name)
target_image_root = os.path.join(data_root, target_dataset_name)
model_root = os.path.join('saved_models')
cudnn.benchmark = True

lr = 3e-4
batch_size = 128
image_size = 28
n_epoch = 50

# manual_seed = random.randint(1, 10000)
manual_seed = 1
random.seed(manual_seed)
torch.manual_seed(manual_seed)
log_dir = 'runs/mnist_advmnist'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok = True)
logger = SummaryWriter(log_dir)

# load data
def data_load(eps):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset_source = datasets.MNIST(
        root=data_root,
        train=True,
        transform=img_transform,
        download=True
    )

    train_dataloader_source = torch.utils.data.DataLoader(
        dataset=train_dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    train_dataset_target = ADVMNISTLoader(
        data_path=os.path.join(target_image_root, 'train_eps{}.npy'.format(eps)),
        transform=img_transform
    )

    train_dataloader_target = torch.utils.data.DataLoader(
        dataset=train_dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    test_dataset_source = datasets.MNIST(
        root=data_root,
        train=False,
        transform=img_transform,
        download=True
    )

    test_dataloader_source = torch.utils.data.DataLoader(
        dataset=test_dataset_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

    test_dataset_target = ADVMNISTLoader(
        data_path=os.path.join(target_image_root, 'test_eps{}.npy'.format(eps)),
        transform=img_transform
    )

    test_dataloader_target = torch.utils.data.DataLoader(
        dataset=test_dataset_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

    return train_dataloader_source, train_dataloader_target, test_dataloader_source, test_dataloader_target


# load model
model = DANNModel()

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

model = model.cuda()
loss_class = loss_class.cuda()
loss_domain = loss_domain.cuda()


def train_one_epoch(model, dataloader_source, dataloader_target, epoch):
    model.train()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        s_img, s_label = data_source_iter.next()
        s_img = s_img.expand(s_img.data.shape[0], 3, 28, 28)

        s_batch_size = s_img.shape[0]
        s_domain_label = torch.zeros(s_batch_size)
        s_domain_label = s_domain_label.long()

        s_img = s_img.cuda()
        s_label = s_label.cuda()
        s_domain_label = s_domain_label.cuda()

        # training model using target data
        t_img, t_label = data_target_iter.next()
        t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)

        t_batch_size = t_img.shape[0]
        t_domain_label = torch.ones(t_batch_size)
        t_domain_label = t_domain_label.long()

        t_img = t_img.cuda()
        t_label = t_label.cuda()
        t_domain_label = t_domain_label.cuda()

        cat_img = torch.cat((s_img, t_img), 0)
        class_output, domain_output = model(input_data=cat_img, alpha=alpha)
        # s_class_output, s_domain_output = model(input_data=s_img, alpha=alpha)

        s_class_output = class_output[:s_batch_size]
        s_domain_output = domain_output[:s_batch_size]
        t_domain_output = domain_output[s_batch_size:]
        t_class_output = class_output[s_batch_size:]

        err_s_label = loss_class(s_class_output, s_label)
        err_t_label = loss_class(t_class_output, t_label)

        err_s_domain = loss_domain(s_domain_output, s_domain_label)
        # _, t_domain_output = model(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(t_domain_output, t_domain_label)

        err = err_t_domain + err_s_domain + err_s_label

        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        # log to tensorboard
        global_step = epoch * (len_dataloader-1) + i
        logger.add_scalar('src_cls_loss', err_s_label.item(), global_step)
        logger.add_scalar('tgt_cls_loss', err_t_label.item(), global_step)
        logger.add_scalar("src_DANN_loss", err_s_label.item()+err_s_domain.item(), global_step)

        i += 1

        if i % 100 == 0:
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))


def test(model, dataloader, dataset_name, epoch):
    alpha = 0

    """ training """
    model = model.eval()
    model = model.cuda()

    # i = 0
    n_total = 0
    n_correct = 0

    for t_img, t_label in dataloader:
        batch_size = t_img.shape[0]
        t_img = t_img.expand(t_img.data.shape[0], 3, 28, 28)
        t_img = t_img.cuda()
        t_label = t_label.cuda()

        class_output, _ = model(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--eps', default=0.3, type=float, help='eps')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)

    train_dataloader_source, train_dataloader_target, \
    test_dataloader_source, test_dataloader_target = data_load(eps=args.eps)

    # training
    for epoch in range(n_epoch):
        train_one_epoch(model, train_dataloader_source, train_dataloader_target, epoch)
        test(model, test_dataloader_source, source_dataset_name, epoch)
        test(model, test_dataloader_target, target_dataset_name, epoch)

    torch.save(model, '{0}/mnist_advmnist.pth'.format(model_root))
