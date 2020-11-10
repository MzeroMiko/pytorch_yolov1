import os
import cv2
cv2.setNumThreads(0) # IMPORTANT! if not zero, may deadlock with dataloader's multiThread
import sys
import time
import json
import random
import torch
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# used only by imageShow
import io
from visdom import Visdom
import matplotlib as mpl 
mpl.use('Agg')

# write image or paint image
class imageShow():

    def __init__(self):
        pass

    # show image with np.array boxes
    @staticmethod
    def boxImageShow(cv2_image, labels, boxes, class_list=[], write_file='', use_visdom=True, resize=False):
        h, w, _ = cv2_image.shape
        if resize:
            boxes[:,0] = boxes[:,0] * w
            boxes[:,1] = boxes[:,1] * h
            boxes[:,2] = boxes[:,2] * w
            boxes[:,3] = boxes[:,3] * h

        for i in range(len(labels)):
            pt1 = (int(boxes[i,0]), int(boxes[i,1]))
            pt2 = (int(boxes[i,2]), int(boxes[i,3]))
            if boxes[i,4] == 0 or pt1 == pt2:
                continue
            color = [0,0,0]
            label = class_list[int(labels[i])] if class_list != [] else str(label[i])
            title = str(round(boxes[i,4], 3)) + ', ' + label
            cv2.rectangle(img=cv2_image, pt1=pt1, pt2=pt2, color=color, thickness=2)
            cv2.putText(img=cv2_image, text=title, org=(pt1[0], pt1[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
        
        if write_file != '':
            cv2.imwrite(write_file, cv2_image)
        if use_visdom:
            img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
            img = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
            Visdom(env='main').image(img, win='showBox')

    @staticmethod
    def curveShow(x_list, y_list, xlabel='', ylabel='', title='', use_visdom=True, write_file=''):
        plt = mpl.pyplot
        plt.plot(x_list, y_list) 
        plt.xlabel(xlabel, size=13)
        plt.ylabel(ylabel, size=13)
        plt.title(title, size=13)  
        
        # turn plt into cv2
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        plt.close()
        buf.seek(0) # to the first address
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        buf.close()
    
        if write_file != '':
            cv2.imwrite(write_file, img)
        if use_visdom:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
            Visdom(env='main').image(img, win='showBox')


# image: cv2_image, objects: np.array([[xmin, ymin, xmax, ymax, label],...])
class imageTransform():

    # return origin BGR image or with random list change 
    @classmethod
    def randomTransform(cls, image, objects=np.array([])):
        image, objects = cls.randomScale(image, objects)
        image = cls.randomBlur(image)
        image = cls.randomHSV(image, 'v')
        image = cls.randomHSV(image, 'h')
        image = cls.randomHSV(image, 's')
        image, objects = cls.randomShift(image, objects)
        image, objects = cls.randomCrop(image, objects)
        return image, objects

    # return origin BGR image or with HSV change 
    @staticmethod
    def randomHSV(image, item='v'):
        if random.random() < 0.5:
            adjustRange = [0.5, 1.5]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if item == 'h':
                h = np.clip(h * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            elif item == 's':
                s = np.clip(s * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            elif item == 'v':
                v = np.clip(v * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        return image

    # return origin BGR image or blurred image 
    @staticmethod
    def randomBlur(image):
        if random.random() < 0.5:
            image = cv2.blur(image, (5,5))
        return image

    # return origin BGR image or with darken and noise  
    @staticmethod
    def randomNoise(image, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            image = image * alpha + random.randrange(-delta, delta)
            image = image.clip(min=0, max=255).astype(np.uint8)
        return image

    # return origin BGR image or fliped left and right (mirror image)
    @staticmethod
    def randomFlip(image, objects=np.array([])):
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            _, w, _ = image.shape
            if objects.shape[0] != 0:
                xmin = w - objects[:,2]
                xmax = w - objects[:,0]
                objects[:,0] = xmin
                objects[:,2] = xmax
        return image, objects

    # return origin BGR image or with width re scaled 
    @staticmethod
    def randomScale(image, objects=np.array([])):
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width*scale), height))
            if objects.shape[0] != 0:
                objects[:,0] = objects[:,0] * scale
                objects[:,2] = objects[:,2] * scale
        return image, objects

    # return origin BGR image or cropped image
    @staticmethod
    def randomCrop(image, objects=np.array([])):
        if random.random() < 0.5:
            height, width, _ = image.shape
            h = int(random.uniform(0.6*height, height))
            w = int(random.uniform(0.6*width, width))
            x = int(random.uniform(0, width-w))
            y = int(random.uniform(0, height-h))
            image = image[y:y+h, x:x+w, :]
            if objects.shape[0] != 0:
                center = (objects[:, 2:4] + objects[:, 0:2]) / 2.0
                center[:,0] = center[:,0] - x
                center[:,1] = center[:,1] - y
                rows_center_in = (center[:,0] > 0) & (center[:,0] < w) & (center[:,1] > 0) & (center[:,1] < h)
                objects = objects[np.where(rows_center_in)[0],:]
                objects[:,0] = np.minimum(np.maximum(objects[:,0] - x, 0), w)
                objects[:,1] = np.minimum(np.maximum(objects[:,1] - y, 0), h)
                objects[:,2] = np.minimum(np.maximum(objects[:,2] - x, 0), w)
                objects[:,3] = np.minimum(np.maximum(objects[:,3] - y, 0), h)
        return image, objects

    # return origin BGR image or shifted (uncovered area are as default)
    @staticmethod
    def randomShift(image, objects=np.array([]), default_color=[104,117,123]):
        if random.random() < 0.5:
            image_shifted = np.zeros(image.shape, dtype=image.dtype)
            image_shifted[:,:,:] = default_color
            height, width, _ = image.shape
            shift_x = int(random.uniform(-width*0.2, width*0.2))
            shift_y = int(random.uniform(-height*0.2, height*0.2))
            # shift image ------------------------------
            if shift_x >= 0 and shift_y >= 0:
                image_shifted[shift_y:, shift_x:, :] = image[:height-shift_y, :width-shift_x, :]
            elif shift_x >= 0 and shift_y < 0:
                image_shifted[:height+shift_y, shift_x:, :] = image[-shift_y:, :width-shift_x, :]
            elif shift_x < 0 and shift_y >= 0:
                image_shifted[shift_y:, :width+shift_x, :] = image[:height-shift_y, -shift_x:, :]
            elif shift_x < 0 and shift_y < 0:
                image_shifted[:height+shift_y, :width+shift_x, :] = image[-shift_y:, -shift_x:, :]
            image = image_shifted
            # shift target ------------------------------
            if objects.shape[0] != 0:
                center = (objects[:, 2:4] + objects[:, 0:2]) / 2.0
                center[:,0] = center[:,0] + shift_x
                center[:,1] = center[:,1] + shift_y
                rows_center_in = (center[:,0] > 0) & (center[:,0] < width) & (center[:,1] > 0) & (center[:,1] < height)
                objects = objects[np.where(rows_center_in)[0],:]
                inside = [max([shift_x, 0]), max([shift_y, 0]), min([width + shift_x, width]), min([height + shift_y, width])]
                objects[:,0] = np.minimum(np.maximum(objects[:,0] + shift_x, inside[0]), inside[2])
                objects[:,1] = np.minimum(np.maximum(objects[:,1] + shift_y, inside[1]), inside[3])
                objects[:,2] = np.minimum(np.maximum(objects[:,2] + shift_x, inside[0]), inside[2])
                objects[:,3] = np.minimum(np.maximum(objects[:,3] + shift_y, inside[1]), inside[3])
        return image, objects

    # return resized image with normalized / resized target
    @staticmethod
    def imageResize(image, objects=np.array([]), image_height=1, image_width=0, normalize=True):
        h, w, _ = image.shape
        image_width = image_height if image_width == 0 else image_width
        image = cv2.resize(image, (image_width, image_height)) # Attention: witdh x height here
        if not normalize:
            w = float(w) / image_width
            h = float(h) / image_height
        if objects.shape[0] != 0:
            objects[:,0] = objects[:,0] / w
            objects[:,1] = objects[:,1] / h
            objects[:,2] = objects[:,2] / w
            objects[:,3] = objects[:,3] / h
        return image, objects


# dataset, return tensor [(image, objects)], image is in [d,h,w], objects is in [[x1,y1,x2,y2,label],...]
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='VOCdevkit', mode_list=['2007train'], image_size=448, train=True):
        # all {mode} picture ids are in file data_path + '/ImageSets/Main/{mode}.txt'
        # mode can be combinations train val trainval test； 2007， 2012；
        self.data_path = str(data_path) + ('/' if str(data_path)[-1] != '/' else '')
        self.image_path = []
        self.image_objects = []
        self.max_objects_num = 0
        mode_list = [mode_list] if  isinstance(mode_list, str) else mode_list
        for year_mode in mode_list:
            year, mode = year_mode[:4], year_mode[4:]
            with open(self.data_path + 'VOC' + year + '/ImageSets/Main/' + mode + '.txt') as f:
                info_list, self.classes = self.parseXml(self.data_path + 'VOC' + year + '/Annotations/', f.read().splitlines())
                for info in info_list:
                    self.image_path.append(self.data_path + 'VOC' + year + '/JPEGImages/' + info["filename"])
                    self.image_objects.append(info["objects"])
                    self.max_objects_num = max([self.max_objects_num, len(info["objects"])])
        self.image_size = image_size
        self.randomTransform = imageTransform.randomTransform
        self.imageResize = imageTransform.imageResize
        self.train = train

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = cv2.imread(self.image_path[item])
        objects = np.array(self.image_objects[item], dtype=np.float32)        
        image, objects = self.randomTransform(image, objects) if self.train else (image, objects)
        # pytorch pretrained model use RGB while cv2 read image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image, objects = self.imageResize(image, objects, self.image_size)
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        new_objects = np.zeros((self.max_objects_num, 5))
        new_objects[0:objects.shape[0], :] = objects.reshape(-1,5)
        return torch.Tensor(image), torch.Tensor(new_objects)

    # return info_list [{"filename":"", "object":[[x1,y1,x2,y2,label],...]},...]; classes: [...]
    @staticmethod
    def parseXml(anno_path, filelist, spec_difficult=0):
        '''
        voc xml sample: VOC2007/Annotations/009931.xml
        '''
        
        info_list = []
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor']
        anno_path = str(anno_path) + ('/' if str(anno_path)[-1] != '/' else '')
        for image_id in filelist:
            # not that not every picture has even one label, if then, xml is not exist.
            try:
                root = ET.parse(anno_path + image_id + '.xml').getroot() 
                # size = root.find('size')
                name = root.find('filename').text
                # width = int(float(size.find('width').text))
                # height = int(float(size.find('height').text))
                # depth = int(float(size.find('depth').text))
                # segmented = int(float(root.find('segmented').text))
                # if spec_segmented != -1 and segmented != spec_segmented:
                #     return ''
                objects = []
                for obj in root.findall('object'):
                    # tmp = {}
                    bndbox = obj.find('bndbox')
                    # tmp['name'] = obj.find('name').text
                    # tmp['pose'] = obj.find('pose').text
                    # tmp['truncated'] = int(float(obj.find('truncated').text))
                    # tmp['difficult'] = int(float(obj.find('difficult').text))
                    # if spec_difficult != -1 and tmp['difficult'] != spec_difficult:
                    #     continue
                    # objects.append(tmp)
                    xmin = int(float(bndbox.find('xmin').text)) - 1 
                    ymin = int(float(bndbox.find('ymin').text)) - 1
                    xmax = int(float(bndbox.find('xmax').text)) - 1
                    ymax = int(float(bndbox.find('ymax').text)) - 1
                    label = classes.index(obj.find('name').text.lower().strip())
                    objects.append([xmin, ymin, xmax, ymax, label])

                info_list.append({"filename": name, "objects": objects}) 
            except:
                pass

        return info_list, classes


# dataset, return tensor [(image, objects)], image is in [d,h,w], objects is in [[x1,y1,x2,y2,label],...]
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data_path='MSCOCO2017', mode_list=['train'], image_size=448, train=True):
        # all {mode} picture ids are in file data_path + '/annotations/instances_{mode}2017.txt'
        # mode can be combinations train val；
        self.data_path = str(data_path) + ('/' if str(data_path)[-1] != '/' else '')
        self.image_path = []
        self.image_objects = []
        self.max_objects_num = 0
        tmp_classes = {}
        mode_list = [mode_list] if  isinstance(mode_list, str) else mode_list
        for mode in mode_list:
            info_list, class_dict = self.parseJson(self.data_path + 'annotations/instances_' + mode + '2017.json')
            tmp_classes.update(class_dict)
            for info in info_list:
                self.image_path.append(self.data_path + 'images/' + info["filename"])
                self.image_objects.append(info["objects"])
                self.max_objects_num = max([self.max_objects_num, len(info["objects"])])
        self.num_classes = max([ int(id) for id in tmp_classes.keys()]) # 90, id=[1,..,90]=[i+1]
        self.classes = [(tmp_classes[str(i+1)] if str(i+1) in tmp_classes.keys() else str(i)) for i in range(self.num_classes)]
        self.image_size = image_size
        self.randomTransform = imageTransform.randomTransform
        self.imageResize = imageTransform.imageResize
        self.train = train

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = cv2.imread(self.image_path[item])
        objects = np.array(self.image_objects[item], dtype=np.float32)
        if self.train:
            image, objects = self.randomTransform(image, objects)
        # pytorch pretrained model use RGB while cv2 read image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image, objects = self.imageResize(image, objects, self.image_size)
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        new_objects = np.zeros((self.max_objects_num, 5))
        new_objects[0:objects.shape[0], :] = objects.reshape(-1,5)
        return torch.Tensor(image), torch.Tensor(new_objects)

    # return image_dict {"{id}":{"filename":"", "object":[[x1,y1,x2,y2,label],...]},...}; class_dict: {"{id}":"name",...}
    @staticmethod
    def parseJson(filename):
        '''
            for instances_{}.json
            {
                "info": info,
                "licenses": [license],
                "images": [image],
                "annotations": [annotation],
                "categories": [category]
            };
            "images": [{
                "license":3,
                "file_name":"COCO_val2014_000000391895.jpg",
                "coco_url":"http:\\\\mscoco.org\\images\\391895",
                "height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
                "flickr_url":"http:*.jpg",
                "id":391895
            },...]
            "categories": [{
                "supercategory": "vehicle", "id": 2, "name": "bicycle"
            },...]
            "annotations": [{
                "segmentation": [[510.66,......]], "area": 702.1057499999998,"iscrowd": 0 or 1,                                                                                                                               
                "image_id": 289343, "bbox": [x,y,w,h], "category_id": 18, "id": 1768
            },...]
        '''
        data = json.load(open(filename, 'r'))
        image_dict = {}
        class_dict = {}
        invalid = 0

        for image in data["images"]:
            image_id = str(image["id"])
            if image_id not in image_dict.keys():
                image_dict[image_id] = {"filename": str(image["file_name"]), "objects": []}

        for cate in data["categories"]:
            cate_id = str(cate["id"])
            if cate_id not in class_dict.keys():
                class_dict[cate_id] = str(cate["name"])

        for anno in data["annotations"]:
            image_id = str(anno["image_id"])
            if image_id not in image_dict.keys():
                invalid += 1
            else:
                bndbox = anno["bbox"]
                label = int(anno["category_id"]) - 1
                xmin = int(bndbox[0])
                ymin = int(bndbox[1])
                xmax = xmin + int(bndbox[2])
                ymax = ymin + int(bndbox[3])
                image_dict[image_id]["objects"].append([xmin, ymin, xmax, ymax, label])

        if invalid != 0:
            print('invalid annotations', invalid)
        return list(image_dict.values()), class_dict


# evaluation function, only use netEval.eval()
class netTest():

    def __init__(self):
        pass

    @classmethod
    def eval(cls, dataset=None, net=None, targetParser=None, data_load='', data_save='', device_ids=[], paint_PR = False, paint_Pred = False):
        
        if data_load != '':
            with open(data_save, "rb") as f:
                data = pickle.load(f)
            preds_pack, truths_pack, preds_raw, truths_raw = data['preds_pack'], data['truths_pack'], data['preds_raw'], data['truths_raw']
        else:
            device = torch.device(('cuda:' + str(device_ids[0])) if len(device_ids) != 0 else 'cpu')
            preds_pack, truths_pack, preds_raw, truths_raw = cls.predTruths(dataset, targetParser, net, device, paint_Pred)
            # do not save data if the data is loaded
            if data_save != '':
                with open(data_save, "wb") as f:
                    pickle.dump({'preds_pack': preds_pack, 'truths_pack': truths_pack, 'preds_raw': preds_raw, 'truths_raw': truths_raw}, f)

        ap_dict = {}
        pr_curve_dict = cls.getPR(preds_pack, truths_pack)
        for label in list(pr_curve_dict.keys()):
            recall_curve = pr_curve_dict[label]['recall']
            precision_curve = pr_curve_dict[label]['precision']
            ap_dict[label] = cls.getAP(recall_curve, precision_curve)
            print('---class {} ap {}---'.format(label, ap_dict[label]))
            if paint_PR:
                imageShow.curveShow(recall_curve, precision_curve, xlabel='recall', ylabel='precision', title=label + ' P-R curve')
                time.sleep(5)

        mean_ap = np.mean(list(ap_dict.values()))
        print('---map {}---'.format(mean_ap))

    # return ap (area of bellow PR curve)
    @staticmethod
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

    # return pr_curve_dict
    @staticmethod
    def getPR(preds, truths, threshold=0.5):
        # preds: [image_id] = {'label_id':[[x1, y1, x2, y2, probs],...]}
        # truths: [image_id] = {'label_id':[[x1, y1, x2, y2],...]}
        # # preds: [class_index][img_index] = [[x1, y1, x2, y2, probs],...]
        # # truths: [class_index][img_index] = [[x1, y1, x2, y2],...]
        # tp: true positive sample, fn: false nagative sample
        # fp: false positive sample, tn: true nagative sample
        # recall: tp / (tp + fn) , identified positive samples / all truly positive samples
        # precision: tp / (tp + fp) , identified positive samples / all labeled positive samples
        # specificity: tn / (tn + fp), accuracy: (tp + tn) / (tp + tn + fp + fn)
        # ap: average Precision, \\int^1_0 precision d(recall)
        # in This function, we want to calc for every box in every image in every label, whether it's tp or fp

        pr_curve_dict = {}
        pred_label_ids = []
        truth_label_ids = []  
        [pred_label_ids.extend(preds[i].keys()) for i in range(len(preds))]
        [truth_label_ids.extend(truths[i].keys()) for i in range(len(truths))]
        label_ids = list(set(truth_label_ids).intersection(set(pred_label_ids)))

        for label in label_ids: 
            num_pred_positive = 0
            num_truth_positive = 0
            score_tp_fp = [] # [[score, tp, fp],...]
            for i in range(len(truths)):
                pred_boxes = np.array(preds[i].get(label, [])) # pred_obj_num * 5, [x1,y1,x2,y2,prob]
                truth_boxes = np.array(truths[i].get(label, [])) # truth_obj_num * 4 [x1,y1,x2,y2]
                num_pred_positive = num_pred_positive + pred_boxes.shape[0]
                num_truth_positive = num_truth_positive + truth_boxes.shape[0]
                # sort to make box with higher probability first match
                if pred_boxes.shape[0] == 0: continue
                pred_boxes = pred_boxes[np.argsort(pred_boxes[:,4])[::-1], :]
                for j in range(pred_boxes.shape[0]):
                    pred_box = pred_boxes[j,:]
                    if truth_boxes.shape[0] == 0:
                        score_tp_fp.append([float(pred_box[4]), 0, 1])
                        continue
                    # calc ious
                    pred_width_height = pred_box[2:4] - pred_box[0:2]
                    truth_width_heights = truth_boxes[:, 2:4] - truth_boxes[:, 0:2]
                    pred_area = pred_width_height[0] * pred_width_height[1]
                    truth_areas = truth_width_heights[:, 0] * truth_width_heights[:, 1]
                    max_left_tops = np.maximum(pred_box[0:2], truth_boxes[:,0:2])
                    min_right_bottoms = np.minimum(pred_box[2:4], truth_boxes[:,2:4])
                    min_width_heights = np.maximum(0.0, (min_right_bottoms - max_left_tops))
                    intersections = min_width_heights[:, 0] * min_width_heights[:, 1]
                    unions = np.maximum(pred_area + truth_areas - intersections, np.finfo(np.float64).eps)
                    ious = intersections / unions
                    ind = ious.argmax()
                    if ious[ind] > threshold:
                        score_tp_fp.append([float(pred_box[4]), 1, 0])
                        # have been matched by one pred box, so delete it
                        truth_boxes = np.delete(truth_boxes, ind, axis=0)
                    else:
                        score_tp_fp.append([float(pred_box[4]), 0, 1])

            score_tp_fp = np.array(score_tp_fp) 
            score_tp_fp = score_tp_fp[np.argsort(score_tp_fp[:,0])[::-1], :] 
            tp_curve = np.cumsum(score_tp_fp[:,1])
            fp_curve = np.cumsum(score_tp_fp[:,2])
            recall_curve = tp_curve / float(num_truth_positive)
            precision_curve = tp_curve / (tp_curve + fp_curve)
            pr_curve_dict[label] = {'recall':recall_curve, 'precision':precision_curve, 'score':score_tp_fp[:,0]}
            
        return pr_curve_dict

    # return preds_pack truths_pack; 
    @staticmethod
    def predTruths(dataset, targetParser, net, device, paint=False):
        # input net, dataset( index : tensor image, tensor truth [[x1,y1,x2,y2],...])
        # input parseTarget(np.array pred : np.array label_boxes [[x1,y1,x2,y2,prob,label],...]), 
        # note that all truths may contain paddings [0,0,0,0,0]
        net.to(device)
        net.eval()

        # generate prediction
        preds_raw, truths_raw = [], []
        for i in tqdm(range(len(dataset))):
            image, truths = dataset[i]
            with torch.no_grad():
                label_boxes = targetParser(net(image.unsqueeze(0).to(device)).cpu().squeeze(0).numpy())
            preds_raw.append(label_boxes)
            truths_raw.append(truths.numpy())
            if paint:
                image = image.permute(1,2,0).numpy()
                labels = label_boxes[:,5].tolist()
                boxes = label_boxes[:,0:5].tolist()
                imageShow.boxImageShow(image, labels, boxes, dataset.classes, resize=True)
                time.sleep(3)
        
        # pack prediction results and ground truths
        preds_pack, truths_pack = [], []
        for i in range(len(preds_raw)):
            # deal with truths ----------
            tmp_truths = {}
            tmp_truths_raw = truths_raw[i]
            for j in range(tmp_truths_raw.shape[0]):
                if (tmp_truths_raw[j,:] == 0.0).all(): # move out paddings
                    continue
                label_str = dataset.classes[int(tmp_truths_raw[j,4])]
                if label_str in tmp_truths.keys():
                    tmp_truths[label_str].append(tmp_truths_raw[j,0:4].tolist())
                else:
                    tmp_truths[label_str] = [tmp_truths_raw[j,0:4].tolist()]
            truths_pack.append(tmp_truths)
            # deal with preds ----------
            tmp_preds = {}
            tmp_preds_raw = preds_raw[i]
            for j in range(tmp_preds_raw.shape[0]):
                label_str = dataset.classes[int(tmp_preds_raw[j,5])]
                if label_str in tmp_preds.keys():
                    tmp_preds[label_str].append(tmp_preds_raw[j,0:5].tolist())
                else:
                    tmp_preds[label_str] = [tmp_preds_raw[j,0:5].tolist()]
                pass
            preds_pack.append(tmp_preds)

        return preds_pack, truths_pack, preds_raw, truths_raw 


# training a network
class netTrain():

    def __init__(self):
        pass
    
    @staticmethod
    def train(
        train_dataset = None, 
        test_dataset = None, 
        net = None, 
        criterion = None, 
        optimizer = None, 
        epoch_model_save = '',
        model_save = './best_model.pth',
        loss_save = './tmp_loss.pkl',
        learning_rate = 0.001, 
        min_learning_rate = 0.0000001,
        num_workers = 8, 
        batch_size = 24, 
        num_epochs = 64,
        tol_epochs = 3, 
        device_ids = [0,1,2,3] 
        ):

        if torch.cuda.is_available():
            device_all = list(range(torch.cuda.device_count()))
            device_ids = list(set(device_ids).intersection(set(device_all)))
        else:
            device_ids = []
        device = torch.device(('cuda:' + str(device_ids[0])) if len(device_ids) != 0 else 'cpu')
        gettime = lambda : time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        loss_list = [] # loss_list = [{'loss_train':[], 'loss_test':[], 'start': '', 'end': ''},...]
        net = torch.nn.DataParallel(net, device_ids=device_ids) if len(device_ids) > 1 else net
        net.to(device)

        optimizer_default = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        # optimizer_default = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)
        optimizer = optimizer_default if optimizer == None else optimizer

        print('train dataset: %d items, test dataset: %d items' % (len(train_dataset), len(test_dataset)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        iteration = 0
        all_tol_epochs = tol_epochs
        best_test_epoch = -1
        best_test_loss = float('inf')
        for epoch in range(num_epochs):
            if epoch - best_test_epoch > all_tol_epochs:
                if learning_rate < min_learning_rate:
                    break
                learning_rate /= 2
                # learning_rate /= 10
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
            for i, (images, truths) in enumerate(train_loader):
                preds = net(images.to(device))
                loss = criterion(preds, truths, iteration)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_list[epoch]['loss_train'].append(loss.item())
                if (i+1) % 20 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.5f, average_loss: %.5f, time: %s' 
                    %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1), gettime()))
                iteration = iteration + 1

            net.eval()
            validation_loss = 0.0
            for i, (images, truths) in enumerate(test_loader):
                with torch.no_grad():
                    preds = net(images.to(device))
                    loss = criterion(preds, truths, iteration)
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



if __name__ == "__main__":
    import torch
    dataset = VOCDataset(data_path='/home/mzero/Data/DataSet/VOCdevkit/', mode_list=['2012trainval', '2007trainval'], train=True)
    dataset = COCODataset(data_path='/home/mzero/Data/DataSet/MSCOCO2017', train=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for i, (images, truths) in enumerate(loader):
        image, target = images[0,:,:,:].numpy(), truths[0,:,:].numpy()
        labels = target[:,4]
        boxes = target[:,0:5] * 1.0
        boxes[:,4] = np.ones(labels.shape)
        # target = yolov1TargetGenerator.genTarget(448, 448, target)
        # probs, labels, boxes = yolov1TargetGenerator.parseTarget(448, 448, target)
        image = np.transpose(np.array(image, dtype=np.float32), (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        # print(boxes, labels)
        imageShow.boxImageShow(image, labels, np.array(boxes), class_list=dataset.classes, resize=True)
        time.sleep(3)





