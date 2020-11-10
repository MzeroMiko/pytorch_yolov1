import sys
import getopt
import torch

from util_general import VOCDataset, netTrain, netTest

from yolo1_net import resYolo
from yolo1_loss import yoloLoss, targetParser


def train(argv):

    data_path = '/home/mzero/Data/DataSet/VOCdevkit/'
    train_mode = ['2012trainval', '2007trainval']
    test_mode = ['2007test']
    device_ids = [0,1,2,3]
    model_save = './best_model.pth'
    loss_save = './tmp_loss.pkl'
    epoch_model_save = ''
    learning_rate = 0.001
    net = resYolo(target_num=30, pretrained=True)

    try:
        opts, _ = getopt.getopt(argv, "hl:m:t:d:", ['help', 'loss_save=', 'model_save=', 'tmp_model=', 'devices='])
    except getopt.GetoptError:
        print('usage: python *.py train [-l | --loss_save save_file_name] [-m | --model_save save_file_name] [-t | --tmp_model save_file_name] [-d | --devices 0123]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: python *.py train [-l | --loss_save save_file_name] [-m | --model_save save_file_name] [-t | --tmp_model save_file_name] [-d | --devices 0123]')
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

    netTrain.train(
        train_dataset = VOCDataset(data_path=data_path, mode_list=train_mode, train=True), 
        test_dataset = VOCDataset(data_path=data_path, mode_list=test_mode, train=False), 
        net = net,
        criterion = yoloLoss(box_num=2, coord_rate=5.0, noobj_rate=0.5),
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4),
        epoch_model_save = epoch_model_save,
        model_save = model_save,
        loss_save = loss_save,
        learning_rate = learning_rate,
        min_learning_rate = learning_rate / 10000,
        batch_size = 24,
        num_epochs = 64,
        tol_epochs = 2,
        device_ids = device_ids
    )


def test(argv):
    device_ids = []
    data_path = '/home/mzero/Data/DataSet/VOCdevkit/'
    mode_list = ['2007test']
    load_model = ''
    load_data = ''
    data_save = ''
    paint = False

    try:
        opts, _ = getopt.getopt(argv, "hpl:s:m:d:", ['help', 'paint', 'load=', 'save=', 'model=', 'devices='])
    except getopt.GetoptError:
        print('usage: python *.py test [-p | --paint] [-l | --load filename] [-s | --save filename] [-m | --model filename] [-d | --devices 0123]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: python *.py test [-p | --paint] [-l | --load filename] [-s | --save filename] [-m | --model filename] [-d | --devices 0123]')
            sys.exit(2)
        elif opt in ('-p', '--paint'):
            paint = True
        elif opt in ('-l', '--load'):
            load_data = arg
        elif opt in ('-s', '--save'):
            data_save = arg
        elif opt in ('-m', '--model'):
            load_model = arg
        elif opt in ('-d', 'devices='):
            try:
                device_ids = [int(i) for i in arg]
            except:
                pass

    print('load_data={}, data_save={}, model={}, device_ids={}'.format(load_data, data_save, load_model, device_ids))

    netTest.eval(
        dataset = VOCDataset(data_path=data_path, mode_list=mode_list, image_size=448, train=False),
        net = resYolo(target_num=30, model_dict=load_model),
        targetParser = targetParser(box_num=2).parse,
        data_load = load_data,
        data_save = data_save,
        device_ids = device_ids,
        paint_PR = paint,
        paint_Pred = paint
    )


if __name__ == '__main__':

    argv = sys.argv[1:]
    if len(argv) == 0:
        print('usage: python *.py [ train | test ] -h')
    elif argv[0] == 'train':
        train(argv[1:])
    elif argv[0] == 'test':
        test(argv[1:])
    else:
        print('usage: python *.py [ train | test ] -h')



# nohup python -u yolo1_main.py train -l tmp_loss_save.pkl -m tmp_model_save.pth -d 0123 2>&1 >> nohup.txt & 
