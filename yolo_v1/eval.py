import cv2
import sys
import time
import torch
import pickle
import getopt
import numpy as np
from cv2 import cv2 
from tqdm import tqdm

from net import resnet50 as yolonet
from utils import VOC_CLASSES, imageTransform, targetGenerator, imageShow


# return ap (area of bellow PR curve)
def getAP(recall_curve, precision_curve):
    # add point (0,1) (1,0)
    recall_curve = np.concatenate(([0.], recall_curve, [1.]))
    precision_curve = np.concatenate(([0.], precision_curve, [0.]))
    # fix sawtooth (abnormal: precision increases as recall inceases) to step stairs
    for i in range(len(precision_curve) - 1, 0, -1):
        precision_curve[i-1] = max(precision_curve[i-1], precision_curve[i])
    # calc areas bellow 
    ind = np.where(recall_curve[1:] != recall_curve[:-1])[0]
    ap = np.sum((recall_curve[ind + 1] - recall_curve[ind]) * precision_curve[ind + 1])

    return ap


# return mean_ap, ap_list for each class
def boxEval(truths, preds, threshold=0.5):
    '''
    preds: [class_index][img_index] = [[x1, y1, x2, y2, probs],...]
    truths: [class_index][img_index] = [[x1, y1, x2, y2],...]
    tp: true positive sample, fn: false nagative sample
    fp: false positive sample, tn: true nagative sample
    recall: tp / (tp + fn) , identified positive samples / all truly positive samples
    precision: tp / (tp + fp) , identified positive samples / all labeled positive samples
    specificity: tn / (tn + fp), accuracy: (tp + tn) / (tp + tn + fp + fn)
    ap: average Precision, \\int^1_0 precision d(recall)
    '''

    all_labels = len(truths)    # 20
    all_images = len(truths[0])
    pr_curve = [] # pr_curve[label] = {'recall':[], 'precision':[], 'score':[]}
    ap_list = -1 * np.ones(all_labels) # ap_list[label] = -1 means no box in this label
    
    for i in range(all_labels):
        pred = preds[i]
        truth = truths[i]
        num_pred_positive = np.sum([len(pred[j]) for j in range(all_images)])
        num_truth_positive = np.sum([len(truth[j]) for j in range(all_images)])
        if num_pred_positive == 0:
            if num_truth_positive != 0:
                ap_list[i] = 0
            continue

        # scan pred as positive samples to get tp and fp
        tp = np.zeros(int(num_pred_positive))
        fp = np.zeros(int(num_pred_positive))
        score = np.zeros(int(num_pred_positive))
        
        count = 0
        for j in range(all_images):
            if len(pred[j]) == 0:
                # no positive prediction found in this label, image
                continue
            truth_boxes = truth[j]
            probs = np.array([float(x[4]) for x in pred[j]])
            boxes = np.array([x[0:4] for x in pred[j]])
            # sort to make box with higher probability first match
            ind = np.argsort(probs)[::-1]
            probs = probs[ind]
            boxes = boxes[ind, :]
            for k in range(len(pred[j])):
                # match by iou one by one
                if len(truth_boxes) != 0:
                    pred_box = np.array(boxes[k])
                    truth_box = np.array(truth_boxes)
                    pred_width_height = pred_box[2:4] - pred_box[0:2]
                    truth_width_heights = truth_box[:, 2:4] - truth_box[:, 0:2]
                    pred_area = pred_width_height[0] * pred_width_height[1]
                    truth_areas = truth_width_heights[:, 0] * truth_width_heights[:, 1]
                    max_left_tops = np.maximum(pred_box[0:2], truth_box[:,0:2])
                    min_right_bottoms = np.minimum(pred_box[2:4], truth_box[:,2:4])
                    min_width_heights = np.maximum(0.0, (min_right_bottoms - max_left_tops))
                    intersections = min_width_heights[:, 0] * min_width_heights[:, 1]
                    ious = intersections / (pred_area + truth_areas - intersections)
                    ind = ious.argmax()
                    if ious[ind] > threshold:
                        tp[count] = 1
                        del truth_boxes[ind] # have been matched by one pred box, so delete it
                fp[count] = 1 - tp[count]
                score[count] = probs[k]
                count += 1

        ind = np.argsort(score)[::-1]
        score, fp, tp = score[ind], fp[ind], tp[ind]
        fp_curve, tp_curve = np.cumsum(fp), np.cumsum(tp)
        recall_curve = tp_curve / float(num_truth_positive)
        precision_curve = tp_curve / (tp_curve + fp_curve)
        pr_curve.append({'recall':recall_curve, 'precision':precision_curve, 'score':score})
        ap_list[i] = getAP(recall_curve, precision_curve)

    mean_ap = np.mean(ap_list[ap_list != -1])
    return mean_ap, ap_list, pr_curve


# return target preds
def predTruths(image_folder, list_file, trained_model, device_ids):
    if torch.cuda.is_available():
        device_all = list(range(torch.cuda.device_count()))
        device_ids = list(set(device_ids).intersection(set(device_all)))
        device_ids = device_all if len(device_ids) == 0 else device_ids
    else:
        device_ids = []
    device = torch.device(('cuda:' + str(device_ids[0])) if len(device_ids) != 0 else 'cpu')

    print('load test data')
    imagefiles, labels, boxes = targetGenerator.loadBoxLabel(list_file)

    all_labels = 20 # 20 means number of classes
    all_images = len(imagefiles)
    preds = [[[]for i in range(all_images)] for i in range(all_labels)]
    truths = [[[]for i in range(all_images)] for i in range(all_labels)]

    print('load trained model') 
    net = yolonet(model_dict=trained_model)  
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    net.to(device)
    net.eval()

    print('prepare ground truth')
    for i in range(all_images):
        for j in range(len(labels[i])):
            truths[labels[i][j]][i].append(boxes[i][j])

    print('prepare predictictions')
    for i in tqdm(range(all_images)):
        image, height, width, _ = imageTransform.loadImageTensor(image_folder + imagefiles[i])
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            pred = net(image).cpu().squeeze(0)
        probs, labels, boxes = targetGenerator.parseTarget(pred, height, width)
        for j in range(len(labels)):
            preds[labels[j]][i].append(boxes[j] + [probs[j]])
        # draw box -------------
        # imageShow.boxImageShow(img, probs, labels, boxes)
        # time.sleep(3)

    return truths, preds


if __name__ == '__main__':

    device_ids = [0]
    image_folder = './allimgs/'
    list_file = 'voc2007test.txt'
    load_model = ''
    load_data = ''
    data_save = ''
    paint = False

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hpl:s:m:d:", ['help','paint', 'load=', 'save=', 'model=', 'devices='])
    except getopt.GetoptError:
        print('usage: python *.py [-p | --paint] [-l | --load filename] [-s | --save filename] [-m | --model filename] [-d | --devices 0123]')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: python *.py [-p | --paint] [-l | --load filename] [-s | --save filename] [-m | --model filename] [-d | --devices 0123]')
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

    def saveData(truths, preds, data_save):
        with open(data_save, "wb") as f:
            pickle.dump({'truths': truths, 'preds': preds}, f)

    def loadData(data_save):
        with open(data_save, "rb") as f:
            data = pickle.load(f)
        return data['truths'], data['preds']
    
    if load_data != '':
        truths, preds = loadData(load_data)
    else:
        truths, preds = predTruths(image_folder, list_file, load_model, device_ids)

    if load_data == '' and data_save != '':
        saveData(truths, preds, data_save)

    mean_ap, ap_list, pr_curve = boxEval(truths, preds)
    for i in range(len(ap_list)):
        if ap_list[i] != -1:
            print('---class {} ap {}---'.format(VOC_CLASSES[i], ap_list[i]))
    print('---map {}---'.format(mean_ap))

    if paint:
        for i in range(len(pr_curve)):
            score = pr_curve[i]['score']
            recall_curve = pr_curve[i]['recall']
            precision_curve = pr_curve[i]['precision']
            imageShow.curveShow(recall_curve, precision_curve, xlabel='recall', ylabel='precision', title=VOC_CLASSES[i] + ' P-R curve')
            time.sleep(5)
