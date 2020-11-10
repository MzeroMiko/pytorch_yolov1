# almost done
import sys
import time
import torch
import pickle
import getopt
import torch.utils.data as data
from torchvision import models

from net import resnet50 as yolonet
from loss import yoloLoss
from utils import yoloDataset

# dataLoader
def datasetLoader(root, list_file, train, batch_size):
    dataset = yoloDataset(root=root, list_file=list_file, train=train)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, len(dataset)


# train the network and save informations
def train(
    device_ids=[0,1,2,3],
    image_folder = './allimgs/',
    train_file = ['voc2012.txt', 'voc2007.txt'],
    test_file = ['voc2007test.txt'],
    epoch_model_save = '',
    model_save = './best_model.pth',
    loss_save = './tmp_loss.pkl',
    learning_rate = 0.001,
    min_learning_rate = 0.0000001,
    num_epochs = 64,
    tol_epochs = 4,
    batch_size = 24,
    grid_num = 14,
    box_num = 2,
    label_num = 20,
    coord_rate = 5.0,
    noobj_rate = 0.5,
    ):
    
    if torch.cuda.is_available():
        device_all = list(range(torch.cuda.device_count()))
        device_ids = list(set(device_ids).intersection(set(device_all)))
        device_ids = device_all if len(device_ids) == 0 else device_ids
    else:
        device_ids = []
    device = torch.device(('cuda:' + str(device_ids[0])) if len(device_ids) != 0 else 'cpu')

    loss_list = [] # loss_list = [{'loss_train':[,...,], 'loss_test':[validation_loss], 'start': '', 'end': ''},...]
    gettime =  lambda :time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

    print('load pre-trained model')
    net = yolonet(pretrained=True)
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    net.to(device)

    print('load loss function')
    criterion = yoloLoss(grid_num, box_num, label_num, coord_rate, noobj_rate)

    print('load optimizer')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    train_loader, train_size = datasetLoader(image_folder, train_file, True, batch_size)
    test_loader, test_size = datasetLoader(image_folder, test_file, False, batch_size)
    print('train dataset has %d images' % (train_size))
    print('test dataset has %d images' % (test_size))

    print('start training ...')
    all_tol_epochs = tol_epochs
    best_test_epoch = -1
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        if epoch - best_test_epoch > all_tol_epochs:
            if learning_rate < min_learning_rate:
                break
            learning_rate /= 10
            all_tol_epochs += tol_epochs
        else:
            all_tol_epochs = tol_epochs
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\nstart epoch [%d/%d], start time : %s' % (epoch + 1, num_epochs, gettime()))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        loss_list.append({'loss_train':[], 'loss_test': [], 'start': gettime(), 'end': ''})

        net.train()
        total_loss = 0.0
        for i,(images,truths) in enumerate(train_loader):
            images = images.to(device)
            truths = truths.to(device)        
            preds = net(images)
            loss = criterion(preds, truths)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list[epoch]['loss_train'].append(loss.item())
            if (i+1) % 20 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.5f, average_loss: %.5f, time: %s' 
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1), gettime()))

        net.eval()
        validation_loss = 0.0
        for i,(images,truths) in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(device)
                truths = truths.to(device)
                preds = net(images)
                loss = criterion(preds,truths)
                validation_loss += loss.item()
        validation_loss /= len(test_loader)
        
        loss_list[epoch]['loss_test'].append(validation_loss)
        if epoch_model_save != '':
            torch.save(net.state_dict(), epoch_model_save)
        if  validation_loss < best_test_loss:
            best_test_epoch = epoch
            best_test_loss = validation_loss
            torch.save(net.state_dict(), model_save)

        loss_list[epoch]['end'] = gettime()
        with open(loss_save, "wb") as f:
            pickle.dump(loss_list, f)
        print('Result: Epoch [%d/%d], validation_loss: %.5f, end time : %s' % (epoch+1, num_epochs, validation_loss, gettime()))
        print('Result: Epoch [%d/%d], best validation loss %.5f' % (best_test_epoch+1, num_epochs, best_test_loss))



if __name__ == '__main__':

    device_ids = [0,1,2,3]
    epoch_model_save = ''
    model_save = './test_model_save.pth'
    loss_save = './test_loss_save.pkl'

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hl:m:t:d:", ['help', 'loss_save=', 'model_save=', 'tmp_model=', 'devices='])
    except getopt.GetoptError:
        print('usage: python *.py [-l | --loss_save save_file_name] [-m | --model_save save_file_name] [-t | --tmp_model save_file_name] [-d | --devices 0123]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: python *.py [-l | --loss_save save_file_name] [-m | --model_save save_file_name] [-t | --tmp_model save_file_name] [-d | --devices 0123]')
            sys.exit(2)
        elif opt in ('-l', '--loss_save'):
            loss_save = arg
        elif opt in ('-m', '--model_save'):
            model_save = arg
        elif opt in ('-t', '--tmp_model'):
            epoch_model_save = arg
        elif opt in ('-d', 'devices='):
            try:
                device_ids = [int(i) for i in arg]
            except:
                pass
    print('loss_save={}, model_save={}, epoch_model_save={}, device_ids={}'.format(loss_save, model_save, epoch_model_save, device_ids))

    train(device_ids=device_ids, loss_save=loss_save, model_save=model_save, epoch_model_save=epoch_model_save,num_epochs = 72)

