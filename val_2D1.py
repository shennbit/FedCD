import numpy as np
import torch
from medpy import metric
from utils import losses
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        #dice_loss = losses.DiceLoss(2)
        #dice = dice_loss(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[480, 640]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    #print(slice.shape)
    x, y = slice.shape[0], slice.shape[1]
    #slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    
    #input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    if len(slice.shape)==2:
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
    else:
        input = torch.from_numpy(slice).unsqueeze(
            0).float().cuda()
    
    #print(input.shape)
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.sigmoid(
            net(input)), dim=1).squeeze(0)
        # out = torch.argmax(torch.softmax(
        #     net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out#zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume1(image, label, net, classes, patch_size=[480, 640]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    # print(slice.shape)
    x, y = slice.shape[0], slice.shape[1]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

    # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    if len(slice.shape) == 2:
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
    else:
        input = torch.from_numpy(slice).unsqueeze(
            0).float().cuda()

    # print(input.shape)
    net.eval()
    with torch.no_grad():
        re = net(input)
        out = torch.argmax(torch.sigmoid(
            re), dim=1).squeeze(0)
        # out = torch.argmax(torch.softmax(
        #     net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out  # zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume2(image, label, net, classes, patch_size=[480, 640]):
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    #slice = image
    # print(slice.shape)
    #x, y = slice.shape[0], slice.shape[1]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

    # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    # if len(slice.shape) == 2:
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).unsqueeze(0).float().cuda()
    # else:
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).float().cuda()

    # print(input.shape)
    input = image
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.sigmoid(
            net(input)), dim=1).squeeze(0)
        # out = torch.argmax(torch.softmax(
        #     net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out  # zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
    
def test_single_volume3(image, pred_b, label, net, classes, patch_size=[480, 640]):
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    #slice = image
    # print(slice.shape)
    #x, y = slice.shape[0], slice.shape[1]
    # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

    # input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    # if len(slice.shape) == 2:
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).unsqueeze(0).float().cuda()
    # else:
    #     input = torch.from_numpy(slice).unsqueeze(
    #         0).float().cuda()

    # print(input.shape)
    input = image
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.sigmoid(
            net(input, pred_b)), dim=1).squeeze(0)
        # out = torch.argmax(torch.softmax(
        #     net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out  # zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[480, 640]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    #slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    #input = torch.from_numpy(slice).unsqueeze(
    #    0).unsqueeze(0).float().cuda()
    input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        output_main, _, _, _ = net(input)
        out = torch.argmax(torch.softmax(
            output_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out#zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list