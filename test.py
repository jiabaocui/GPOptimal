import torch
import torch.nn as nn
import torchvision
import datetime
import  time
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
from numpy.linalg import inv
import torch.nn.functional as F
from meters import flush_scalar_meters
from fn_get_meters import get_meters
from resnet_model_cifar10_GP import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
#from resnext import resnext29_8_64

#choose model
train_meters = get_meters('train')
val_meters = get_meters('val')
topk = [1, 5]
dict_own = {}
batch_size = 128
val_batch_size = 70
num_class = 100
GP_num = 70

model = torch.nn.DataParallel(resnet20(100)).cuda()
criterion = nn.CrossEntropyLoss()
criterion1 = nn.KLDivLoss(reduction="batchmean")
use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9, weight_decay=0.0001)
model_dict = model.state_dict()

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

def own_KL(input0, label0):
    input = F.softmax(input0, dim=1)
    label = F.softmax(label0, dim=1)
    result_KL = criterion1(input, label)
    return result_KL

def GP_new_cal(input):
    batch_size,_ = input.shape
    sum = torch.rand((batch_size, batch_size))
    for j in range(batch_size):
        sum[j, :] = (2.718281828 ** (
                    -(1 / GP_num) * torch.norm(((input[j, :].expand_as(input)) - input), 2, 1))).cpu().detach()
    #    print("sum: ", sum)
    K_ni = torch.from_numpy(inv(sum)).cuda().detach()
#    print(sum)
    return sum, K_ni


def GP_loss_use_cal(old_data_use, K_ni, input):
    B, _ = input.shape
    old_data_use1 = old_data_use.detach().cuda()
    total_num,_ = old_data_use1.shape
    sum_here = torch.rand((B, total_num)).cuda()
    for i in range(B):
        sum_here[i, :] = 2.718281828 ** (
                    -(1 / GP_num) * torch.norm(((input[i, :].expand_as(old_data_use1)) - old_data_use1), 2, 1))
    K_a = sum_here
    K_u = K_a.mm(K_ni)
    K_var_sum = 0
    K_var = -(K_a.mm(K_ni)).mm(torch.t(K_a)) + 1
    for i in range(B):
        K_var_sum = K_var_sum + float(K_var[i, i])
    K_var_sum = K_var_sum / B
#    print("K_var: ", K_var_sum)
    return K_u, K_var_sum, B

def data_loader_train():
    img_data = torchvision.datasets.CIFAR100("./100_data", train=True, transform=transform_train, target_transform=None,
                                             download=True)

    print('train_total:', len(img_data))
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=batch_size,shuffle=True,num_workers=12)
    print('train_batch_total：', len(data_loader))
    return data_loader

def data_loader_test():
    img_data = torchvision.datasets.CIFAR100("./100_data", train=False, transform=transform_test, target_transform=None,
                                             download=True)

    print('val_total:', len(img_data))
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=batch_size,shuffle=False,num_workers=12)
    print('val_batch_total：', len(data_loader))
    return data_loader

def data_loader_val():
    img_data = torchvision.datasets.ImageFolder('./cifar100_val_all', transform=transform_val)

    print('val_total:', len(img_data))
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=val_batch_size,shuffle=False,num_workers=12)
    print('val_batch_total：', len(data_loader))
    return data_loader




"""
pretrained_dict_mobile = torch.load('./checkpoint/ckpt_top5_99.pth')
x = pretrained_dict_mobile["net"]
for key1, value in x.items():
    key = str('module.') + key1
    dict_own.update({key: value})

pretrained_dict = {k: v for k, v in dict_own.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)
"""

data_train = data_loader_train()
data_test = data_loader_test()
data_val = data_loader_val()

def val_data_build():
    model.eval()
    start_time = time.time()
    base_data = []
    base_target = []
    use_tensor = []
    use_target = []
    use_K_ni = []
    pretrained_dict_mobile = torch.load('./checkpoint/5near_list_local_5.pth')
    class_np = pretrained_dict_mobile["20near_list"]
    for i, (batch_x, batch_y) in enumerate(data_val):
        batch_x = batch_x.cuda()
#        batch_y = batch_y.cuda()
        one_hot_old = batch_y.unsqueeze(1)
        one_hot = torch.zeros((val_batch_size, num_class)).scatter_(1, one_hot_old, 1)
        one_hot = one_hot.cuda()
        output, features = model(batch_x)
        base_data.append(np.array(features.cpu().detach()))
        base_target.append(np.array(one_hot.cpu().detach()))
    for k in range(num_class):
 #       print('class_np.shape: ', class_np.shape)
        _, long = class_np.shape
        for i in range(long):
            base_data_tensor1 = base_data[class_np[k][i]]
            base_data_target1 = base_target[class_np[k][i]]
            if i == 0:
                base_data_tensor = base_data_tensor1
                base_data_target = base_data_target1
            else:
                base_data_tensor = np.append(base_data_tensor, base_data_tensor1, axis=0)
                base_data_target = np.append(base_data_target, base_data_target1, axis=0)
#        print("Here is: ", base_data_tensor.shape)
        base_data_tensor = torch.from_numpy(base_data_tensor).cuda()
        base_data_target = torch.from_numpy(base_data_target).cuda()
        _, K_ni = GP_new_cal(base_data_tensor)
        use_tensor.append(base_data_tensor)
        use_target.append(base_data_target)
        use_K_ni.append(K_ni)
    end_time = time.time()
    print('build_val_time: ', end_time - start_time)
#    print('use_target: ', len(use_target), use_target[0].shape)
    return use_tensor, use_target, use_K_ni

def train(epoch,k_loss2, k_loss3):
    start_time = time.time()
    print('\nEpoch: %d' % epoch)
    model.train()
    i_num = 0
    loss_show = 0
    top1 = 0
    err = 0.99

    val_tensor_list, val_target_list, val_K_ni_list = val_data_build()
    for i, (batch_x, batch_y) in enumerate(data_train):

        if i % 200 == 0:
            err, loss_test = test(epoch, i)
            f1 = open('./test_err_2GP_loss1_32.txt', 'r+')
            f1.read()
            f1.write('\n')
            f1.write(str(loss_test))
            f1.close()
            f2 = open('./test_err_2GP_err_32.txt', 'r+')
            f2.read()
            f2.write('\n')
            f2.write(str(err))
            f2.close()
        i_num = i_num + 1

        model.train()
#        if i % 300 ==0:
#            print("iter: ", i)
        optimizer.zero_grad()
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        B_num, _, _, _ = batch_x.shape
        i_num = i_num + 1
        output, features = model(batch_x)
        _, channel = features.shape
        current_num = int(batch_y[0])
        tmp = torch.zeros((1, channel)).cuda()
        tmp[0, :] = features[0, :]
        K_var_all = 0
        K_u, K_var, _ = GP_loss_use_cal(val_tensor_list[current_num].cuda(),val_K_ni_list[current_num].cuda(), tmp)
        K_var_all = K_var_all + K_var
        result_V_T = (K_u.mm(val_target_list[current_num]))
        for j in range(B_num - 1):
            current_num = int(batch_y[0])
            tmp = torch.zeros((1, channel)).cuda()
            tmp[0, :] = features[j + 1, :]
            K_u, K_var, _ = GP_loss_use_cal(val_tensor_list[current_num].cuda(),val_K_ni_list[current_num].cuda(), tmp)
            K_var_all = K_var_all + K_var
            result_V_T_tmp = (K_u.mm(val_target_list[current_num])).detach()
            result_V_T = torch.cat([result_V_T, result_V_T_tmp], 0)
        K_var_avg = K_var_all / B_num
        k3 = 1/(1+K_var_avg)
        result_V_T1 = result_V_T.detach()
        result_V_T2 = result_V_T
        _, y = torch.max(result_V_T, 1)

        loss1 = torch.mean(criterion(output, batch_y))
        loss2 = torch.mean(own_KL(output, result_V_T1))
        loss3 = torch.mean(criterion(result_V_T2, batch_y))
        loss = loss1 / (1 + float(1 - err)) + k3 * float(1 - err) * (k_loss2 * loss2 + k_loss3 * loss3) / (1 + float(1 - err)) / 2
        k_loss_out2 = float(loss1) / float(loss2)
        k_loss_out3 = float(loss1) / float(loss3)
        loss.backward()
        optimizer.step()


        train_meters[str(3 + 1)]['loss'].cache(
            loss.cpu().detach().numpy())
        # topk
        _, pred = output.topk(max(topk))
        pred = pred.t()
        correct = pred.eq(batch_y.view(1, -1).expand_as(pred))
        for k in topk:
            correct_k = correct[:k].float().sum(0)
            error_list = list(1. - correct_k.cpu().detach().numpy())
            train_meters[str(3 + 1)]['top{}_error'.format(k)].cache_list(error_list)


        results = flush_scalar_meters(train_meters[str(3 + 1)])
        top1 = top1 + results["top1_error"]
        loss_show = loss_show + results["loss"]
    print("-------------------")
    print('AVG_TRAIN_LOSS3: ', loss_show / i_num)
    print('AVG_Top1_error3: ', top1 / i_num)
    f3 = open('./train_err_2GP_loss_32.txt', 'r+')
    f3.read()
    f3.write('\n')
    f3.write(str(loss_show / i_num))
    f3.close()
    f4 = open('./train_err_2GP_err_32.txt', 'r+')
    f4.read()
    f4.write('\n')
    f4.write(str(top1 / i_num))
    f4.close()
    end_time = time.time()
    print('this_train_epoch_time: ', end_time - start_time)
    return k_loss2, k_loss3

def test(epoch, i):
    print('\nTest: ', epoch, 'i_num: ', i)
    model.eval()
    i_num = 0
    top1 = 0
    loss1_all = 0
    for i, (batch_x, batch_y) in enumerate(data_test):
        B,_,_,_ = batch_x.shape
        i_num = i_num + 1
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output, _ = model( batch_x)
        loss1 = float(torch.mean(criterion(output, batch_y)).cpu().detach())
        loss1_all = loss1_all + loss1
        _, pred = output.topk(max(topk))
        pred = pred.t()
        correct = pred.eq(batch_y.view(1, -1).expand_as(pred))
        correct_k_list = correct[:1].float().sum(0)
        correct_k = torch.sum(correct_k_list)
        error = 1.0 - (float(correct_k) / int(B))
        top1 = top1 + error
    print( 'OUT_TEST_Top1_error3: ',top1 / i_num)
    return (top1 / i_num), (loss1_all / i_num)

# Save checkpoint.
def save(epoch):
    print('Saving..')
    state = {
        'net': model.module.state_dict(),
        }
    torch.save(state, './checkpoint/5/ckpt_top5_res20_' + str(epoch) + '.pth')


def adjust_learning_rate(optimizer,epoch):
    if epoch == 99:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    if epoch == 149:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


start_time_all = time.time()
k_loss2 = 1
k_loss3 = 1
test(epoch, 0)

