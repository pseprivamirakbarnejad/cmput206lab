import os
import cv2
import sys
import re
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import subprocess

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import Dataset

from A10_modules import Classifier
from A10_modules import TrainParams

from tensorboardX import SummaryWriter


class A10_Params:
    """

    :ivar dataset:
        0: MNIST
        1: FMNIST
    """

    def __init__(self):
        self.use_cuda = 1
        self.enable_test = 0
        self.train = TrainParams()


class FontsDataset(Dataset):
    """
    :param Dataset _dataset:
    """

    def __init__(self, fname='train_data.npz'):
        """

        :param Dataset dataset:
        :param all_idx:
        :param float labeled_percent:
        """

        train_data = np.load(fname, allow_pickle=True)
        self._train_images, self._train_labels = train_data['images'], train_data[
            'labels']  # type: np.ndarray, np.ndarray

        self._train_images = self._train_images.astype(np.float32) / 255.0
        self._train_images = np.expand_dims(self._train_images, axis=1)

        self._train_labels = self._train_labels.astype(np.int64)
        self._n_train = self._train_images.shape[0]

    def __len__(self):
        return self._n_train

    def __getitem__(self, idx):
        assert idx < self._n_train, "Invalid idx: {} for _n_data: {}".format(idx, self._n_train)

        input = self._train_images[idx, ...]
        target = self._train_labels[idx]
        return input, target


def evaluate(classifier, data_loader, criterion_cls, vis, device):
    classifier.eval()

    mean_loss_sum = 0
    _psnr_sum = 0
    total = 0
    correct = 0

    n_batches = 0
    _pause = 1

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = classifier(inputs)

            loss = criterion_cls(outputs, targets)

            mean_loss = loss.item()

            mean_loss_sum += mean_loss

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            n_batches += 1

            # outputs_np = outputs.detach().cpu().numpy()

            if vis:
                inputs_np = inputs.detach().cpu().numpy()
                comcat_imgs = []
                for i in range(data_loader.batch_size):
                    input_img = inputs_np[i, ...].squeeze()
                    target = targets[i]
                    output = predicted[i]

                    output_img = np.zeros_like(input_img)
                    _text = '{}'.format(chr(65 + int(output)))
                    cv2.putText(output_img, _text, (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, 255, 1, cv2.LINE_AA)

                    target_img = np.zeros_like(input_img)
                    _text = '{}'.format(chr(65 + int(target)))
                    cv2.putText(target_img, _text, (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, 255, 1, cv2.LINE_AA)

                    comcat_img = np.concatenate((input_img, target_img, output_img), axis=0)
                    comcat_imgs.append(comcat_img)

                comcat_imgs = np.concatenate(comcat_imgs, axis=1)
                cv2.imshow('comcat_imgs', comcat_imgs)
                k = cv2.waitKey(1 - _pause)
                if k == 27:
                    sys.exit(0)
                elif k == ord('q'):
                    vis = 0
                    cv2.destroyWindow('comcat_imgs')
                    break
                elif k == 32:
                    _pause = 1 - _pause

    overall_mean_loss = mean_loss_sum / n_batches
    acc = 100. * correct / total

    if vis:
        cv2.destroyWindow('comcat_imgs')

    return overall_mean_loss, acc


def main():
    params = A10_Params()

    # optional command line argument parsing
    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    # init device
    if params.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Training on CPU')

    train_params = params.train

    # load dataset
    train_set = FontsDataset()

    num_train = len(train_set)
    indices = list(range(num_train))

    assert train_params.valid_ratio > 0, "Zero validation ratio is not allowed "
    split = int(np.floor((1.0 - train_params.valid_ratio) * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]

    print('Training samples: {}\n'
          'Validation samples: {}\n'
          ''.format(
        len(train_idx),
        len(valid_idx),
    ))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_params.batch_size, sampler=train_sampler,
                                                   num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(train_set, batch_size=24, sampler=valid_sampler,
                                                   num_workers=4)

    # create modules
    classifier = Classifier().to(device)

    assert isinstance(classifier, nn.Module), 'classifier must be an instance of nn.Module'

    classifier.init_weights()

    # create losses
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = classifier.parameters()

    # create optimizer
    if train_params.optim_type == 0:
        optimizer = torch.optim.SGD(parameters, lr=train_params.lr,
                                    momentum=train_params.momentum,
                                    weight_decay=train_params.weight_decay)
    elif train_params.optim_type == 1:
        optimizer = torch.optim.Adam(parameters, lr=train_params.lr,
                                     weight_decay=train_params.weight_decay,
                                     eps=train_params.eps,
                                     )
    else:
        raise IOError('Invalid optim_type: {}'.format(train_params.optim_type))

    # optimizer = torch.optim.Adam(classifier.parameters())

    weights_dir = os.path.dirname(train_params.weights_path)
    weights_name = os.path.basename(train_params.weights_path)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    tb_path = os.path.join(weights_dir, 'tb')
    writer = SummaryWriter(logdir=tb_path)
    print(f'Saving tensorboard summary to: {tb_path}')
    # subprocess.Popen("tensorboard --logdir={}".format(tb_path))
    os.system("tensorboard --logdir={}".format(tb_path))

    start_epoch = 0
    max_valid_acc_epoch = 0
    max_valid_acc = 0
    max_train_acc = 0
    min_valid_loss = np.inf
    min_train_loss = np.inf
    valid_loss = valid_acc = -1
    # load weights
    if train_params.load_weights:
        matching_ckpts = [k for k in os.listdir(weights_dir) if
                          os.path.isfile(os.path.join(weights_dir, k)) and
                          k.startswith(weights_name)]
        if not matching_ckpts:
            msg = 'No checkpoints found matching {} in {}'.format(weights_name, weights_dir)
            if train_params.load_weights == 1:
                raise IOError(msg)
            print(msg)
        else:
            matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            weights_path = os.path.join(weights_dir, matching_ckpts[-1])

            chkpt = torch.load(weights_path, map_location=device)  # load checkpoint

            print('Loading weights from: {} with:\n'
                  '\tepoch: {}\n'
                  '\ttrain_loss: {}\n'
                  '\ttrain_acc: {}\n'
                  '\tvalid_loss: {}\n'
                  '\tvalid_acc: {}\n'
                  '\ttimestamp: {}\n'.format(
                weights_path, chkpt['epoch'],
                chkpt['train_loss'], chkpt['train_acc'],
                chkpt['valid_loss'], chkpt['valid_acc'],
                chkpt['timestamp']))

            classifier.load_state_dict(chkpt['classifier'])
            optimizer.load_state_dict(chkpt['optimizer'])

            max_valid_acc = chkpt['valid_acc']
            min_valid_loss = chkpt['valid_loss']

            max_train_acc = chkpt['train_acc']
            min_train_loss = chkpt['train_loss']

            max_valid_acc_epoch = chkpt['epoch']
            start_epoch = chkpt['epoch'] + 1

    if train_params.load_weights != 1:
        # continue training
        for epoch in range(start_epoch, train_params.n_epochs):
            # Training
            classifier.train()

            train_loss = 0
            train_total = 0
            train_correct = 0
            batch_idx = 0

            save_weights = 0

            for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = classifier(inputs)
                outputs_np = outputs.detach().cpu().numpy()

                loss = criterion(outputs, targets)

                mean_loss = loss.item()
                train_loss += mean_loss

                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            mean_train_loss = train_loss / (batch_idx + 1)

            train_acc = 100. * train_correct / train_total

            writer.add_scalars('data/scalar_group', {
                'train_loss': mean_train_loss,
                'train_acc': train_acc,
            }, epoch)

            if epoch % train_params.valid_gap == 0:

                valid_loss, valid_acc = evaluate(
                    classifier, valid_dataloader, criterion, train_params.vis, device)

                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    max_valid_acc_epoch = epoch
                    if train_params.save_criterion == 0:
                        save_weights = 1

                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    if train_params.save_criterion == 1:
                        save_weights = 1

                if train_acc > max_train_acc:
                    max_train_acc = train_acc
                    if train_params.save_criterion == 2:
                        save_weights = 1

                if train_loss < min_train_loss:
                    min_train_loss = train_loss
                    if train_params.save_criterion == 3:
                        save_weights = 1

                writer.add_scalars('data/scalar_group', {
                    'train_loss': mean_train_loss,
                    'train_acc': train_acc,
                }, epoch)

                print(
                    'Epoch: %d Train-Loss: %.6f  | Train-Acc: %.3f%% | '
                    'Validation-Loss: %.6f | Validation-Acc: %.3f%% | '
                    'Max Validation-Acc: %.3f%% (epoch: %d)' % (
                        epoch, mean_train_loss, train_acc,
                        valid_loss, valid_acc, max_valid_acc, max_valid_acc_epoch))

            # Save checkpoint.
            if save_weights:
                model_dict = {
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': mean_train_loss,
                    'train_acc': train_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S"),
                }
                weights_path = '{}.{:d}'.format(train_params.weights_path, epoch)
                print('Saving weights to {}'.format(weights_path))
                torch.save(model_dict, weights_path)

        if params.enable_test:
            test_set = FontsDataset('test_data.npz')
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=24, num_workers=4)
        else:
            test_dataloader = valid_dataloader

        start_t = time.time_ns()
        _, test_acc = evaluate(
            classifier, test_dataloader, criterion, train_params.vis, device)
        end_t = time.time_ns()
        test_time = end_t - start_t

        print('test_acc: {:.4f}'.format(test_acc))
        print('test_time: {:.4f}'.format(test_time))


if __name__ == '__main__':
    main()
