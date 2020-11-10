import os
import io
import cv2
import sys
import time
import pickle
import getopt
import torch
import random
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

from cv2 import cv2
from visdom import Visdom
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]


# write image or paint image
class imageShow():

    def __init__(self):
        pass

    # show image with boxes
    @staticmethod
    def boxImageShow(cv2_image, probs, labels, boxes, write_file='', use_visdom=True, resize=False):
        h, w, _ = cv2_image.shape
        boxes = torch.Tensor(boxes)
        if resize:
            boxes = boxes * torch.Tensor([w, h, w, h]).expand_as(boxes)

        for i in range(len(probs)):
            pt1 = (int(boxes[i][0]), int(boxes[i][1]))
            pt2 = (int(boxes[i][2]), int(boxes[i][3]))
            if probs == 0 or pt1 == pt2:
                continue
            color = Color[int(labels[i])]
            title = str(round(probs[i], 3)) + ', ' + VOC_CLASSES[int(labels[i])]
            cv2.rectangle(img=cv2_image, pt1=pt1, pt2=pt2, color=color, thickness=2)
            cv2.putText(img=cv2_image, text=title, org=(pt1[0], pt1[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
        
        if write_file != '':
            cv2.imwrite(write_file, cv2_image)
        if use_visdom:
            img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) 
            img = transforms.ToTensor()(img)
            Visdom(env='main').image(img, win='showBox')

    @staticmethod
    def curveShow(x_list, y_list, xlabel='', ylabel='', title='', use_visdom=True, write_file=''):
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
            img = transforms.ToTensor()(img)
            Visdom(env='main').image(img, win='showBox')


# function class generating and parsing target
class targetGenerator():

    def __init__(self):
        pass

    # return name x1 y1 x2 y2 class x1 y1 x2 y2 class ... 
    @staticmethod
    def parseVOCXml(filename, spec_difficult=0):
        '''
        voc xml sample: VOC2007/Annotations/009931.xml:
        <annotation>
            <folder>VOC2007</folder>
            <filename>009931.jpg</filename>
            <source>
                <database>The VOC2007 Database</database>
                <annotation>PASCAL VOC2007</annotation>
                <image>flickr</image>
                <flickrid>336444880</flickrid>
            </source>
            <owner>
                <flickrid>Lothar Lenz</flickrid>
                <name>Lothar Lenz</name>
            </owner>
            <size>
                <width>500</width>
                <height>332</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            <object>
                <name>person</name>
                <pose>Frontal</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>135</xmin>
                    <ymin>159</ymin>
                    <xmax>170</xmax>
                    <ymax>250</ymax>
                </bndbox>
                <part>
                    <name>head</name>
                    <bndbox>
                        <xmin>147.7376</xmin>
                        <ymin>158.2127</ymin>
                        <xmax>159.3398</xmax>
                        <ymax>173.6823</ymax>
                    </bndbox>
                </part>
                <part>
                    ...
                </part>
            </object>
            <object>
                <name>horse</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>90</xmin>
                    <ymin>160</ymin>
                    <xmax>143</xmax>
                    <ymax>250</ymax>
                </bndbox>
            </object>
            <object>
                ...
            </object>
        </annotation>
        '''

        root = ET.parse(filename).getroot() 
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
            tmp = {}
            bndbox = obj.find('bndbox')
            tmp['name'] = obj.find('name').text
            tmp['pose'] = obj.find('pose').text
            tmp['truncated'] = int(float(obj.find('truncated').text))
            tmp['difficult'] = int(float(obj.find('difficult').text))
            tmp['xmin'] = int(float(bndbox.find('xmin').text))
            tmp['ymin'] = int(float(bndbox.find('ymin').text))
            tmp['xmax'] = int(float(bndbox.find('xmax').text))
            tmp['ymax'] = int(float(bndbox.find('ymax').text))
            if spec_difficult != -1 and tmp['difficult'] != spec_difficult:
                continue
            objects.append(tmp)
        # print(filename, width, height, depth, segmented, objects)
        info = ''
        for obj in objects:
            info = info + ' ' + str(obj['xmin']) + ' ' + str(obj['ymin'])
            info = info + ' ' + str(obj['xmax']) + ' ' + str(obj['ymax'])
            info = info + ' ' + str(VOC_CLASSES.index(obj['name']))
        if str(name).strip() == '':
            return ''
        return name + info + '\n'

    # from annotation folder generate a text
    @classmethod
    def VOCXml2Txt(cls, annotations, out_file):
        out_file = open(out_file, 'w')
        xml_files = os.listdir(annotations)
        xml_files.sort()
        for xml_file in xml_files:
            out_file.write(cls.parseVOCXml(annotations + xml_file))
        out_file.close()

    # return list imagefile names, labels (0-19), boxes
    @staticmethod
    def loadBoxLabel(list_file):
        imagefiles, labels, boxes = [], [], []
        # fix arg listfile ------------------
        list_file = [list_file] if  isinstance(list_file, str) else list_file
        # read boxes and labels in lines -------------
        lines = []
        for filename in list_file:
            with open(filename) as f:
                lines.extend(f.readlines())
        for line in lines:
            splited = line.strip().split()
            imagefiles.append(splited[0])
            # one image could has many boxes
            box = []
            label = []
            num_box = (len(splited) - 1) // 5
            for i in range(num_box):
                x1 = float(splited[1+5*i])
                y1 = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                box.append([x1, y1, x2, y2])
                label.append(int(splited[5+5*i]))
            boxes.append(box)
            labels.append(label)
        return imagefiles, labels, boxes

    # return target of 14x14x30, 30 means label can be 0-19
    @staticmethod
    def genTarget(height, width, boxes, labels):
        # TARGET: 14x14x30 (in YOLO v1, 7x7x30)
        # target[i,j,0:4]: bounding box 1 position (dx dy w h)
        # target[i,j,4]: confidence for box 1
        # target[i,j,5:9]: bounding box 2 position (dx dy w h)
        # target[i,j,9]: confidence for box 2
        # target[i,j,10:30]: probability for the object with this label if truly have object
        # dx = center_x / cell_size - i; dy = center_y / cell_size - j
        # confidence for box k = P{have object} * IOU{prediction box compare to ground truth}
        # IOU{prediction box compare to ground truth} = 1, when in eval mode
        
        box_num = 2
        grid_num = 14
        target_num = 30
        target = torch.zeros((grid_num, grid_num, target_num))
        cell_size = 1.0 / grid_num
        boxes = boxes / torch.Tensor([width, height, width, height]).expand_as(boxes)
        box_size = boxes[:,2:] - boxes[:,:2]
        center = (boxes[:,2:] + boxes[:,:2]) / 2.0
        # if in one grid, then cover the target
        for i in range(center.size()[0]):
            grid = (center[i] / cell_size).ceil() - 1 # center in which grid
            delta = center[i] / cell_size - grid # delta between center and grid_left_top
            coordx, coordy = int(grid[0]), int(grid[1]) # since cv2_image is h,w,c, so coordy first
            target[coordy, coordx, int(labels[i]) + box_num * 5] = 1
            for b in range(box_num):
                tmp_ind = b * 5
                target[coordy, coordx, tmp_ind:tmp_ind+2] = delta
                target[coordy, coordx, tmp_ind+2:tmp_ind+4] = box_size[i]
                target[coordy, coordx, tmp_ind+4] = 1
        return target

    # non maximum suppression, return list keep whith input np.array boxes and probs 
    @staticmethod
    def nms(boxes, probs, threshold=0.5):
        # input probs: np.array([...]), boxes: np.array([[x1,y1,x2,y2],...])
        keep = []
        order = np.argsort(probs)[::-1]
        left_tops = boxes[:,0:2]
        right_bottoms = boxes[:,2:4]
        width_heights = right_bottoms - left_tops
        areas = width_heights[:,0] * width_heights[:,1]
        while len(order) > 0:
            # store max confidence 
            i = order[0]
            keep.append(i)
            order = order[1:]
            if len(order) == 0: break
            # calc iou = intersection / union
            max_left_tops = np.maximum(left_tops[i,:], left_tops[order,:])
            min_right_bottoms = np.minimum(right_bottoms[i,:], right_bottoms[order,:])
            min_width_heights = np.maximum(0.0, (min_right_bottoms - max_left_tops))
            intersections = min_width_heights[:, 0] * min_width_heights[:, 1]
            ious = intersections / (areas[i] + areas[order] - intersections)
            order = order[np.where(ious <= threshold)[0]]
        
        return keep

    # return list probs, labels(0-19), boxes with input tensor target 14x14x30, use nms
    @classmethod
    def parseTarget(cls, target, height, width, threshold=0.1):
        probs, labels, boxes = [], [], []
        box_num = 2
        grid_num = 14
        all_box_num = 5 * box_num
        cell_size = 1.0 / grid_num
        conf_ind = list(range(4, all_box_num, 5))
        for i in range(grid_num):
            for j in range(grid_num):
                tmp_info = []
                box_target = target[i, j, :all_box_num].numpy()
                box_confs = box_target[conf_ind] 
                prob_label, label = target[i, j, all_box_num:].max(0)
                prob_label, label = prob_label.item(), label.item()
                # check if some boxes has high probabilities
                for b in range(box_num):
                    ind = 5 * b
                    if box_target[ind+4] > threshold:
                        tmp_info.append(tuple(box_target[ind:ind+5]))
                # nothing found so far
                if  len(tmp_info) == 1:
                    ind = box_confs.argmax() * 5
                    tmp_info.append(tuple(box_target[ind:ind+5]))
                # turn tmp_info into [prob, label, box]
                tmp_info = list(set(tmp_info)) # delete repeated value
                for tmp in tmp_info:
                    tmp = list(tmp)
                    center_x, center_y = cell_size * (tmp[0] + j), cell_size * (tmp[1] + i)
                    x1 = (center_x - tmp[2] / 2.0) * width
                    x2 = (center_x + tmp[2] / 2.0) * width
                    y1 = (center_y - tmp[3] / 2.0) * height
                    y2 = (center_y + tmp[3] / 2.0) * height 
                    prob = prob_label * tmp[4]
                    if prob > threshold:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(label))
                        probs.append(prob)
                        
        # non maximum suppression 
        if len(probs) == 0:
            return [], [], [[]]
        probs, labels, boxes = np.array(probs), np.array(labels), np.array(boxes)
        keep = cls.nms(boxes, probs)
        return probs[keep].tolist(), labels[keep].tolist(), boxes[keep].tolist() 


# function class transforing images
class imageTransform():
    
    voc_mean = (123,117,104) # VOC images mean RGB
    image_size = 448 # YOLO input size

    def __init__(self):
        pass

    @classmethod
    def loadImageTensor(cls, filename):
        img = cv2.imread(filename)
        h, w, d = img.shape    
        img = cls.imageTensor(img)
        return img, h, w, d

    # return 3x448x448 tensor image with input cv2 image
    @classmethod
    def imageTensor(cls, img):
        mean = cls.voc_mean
        image_size = cls.image_size
        # pytorch pretrained model use RGB while cv2 read image as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = img - np.array(mean, dtype=np.float32)
        img = cv2.resize(img, (image_size, image_size)) # would interpolate so image is type float
        img = transforms.ToTensor()(img)
        return img

    # return bgr cv2 image while input a tensor image produced by __getitem__()
    @classmethod
    def imageRestore(cls, img, height, width):
        mean = cls.voc_mean
        img = img.clone().detach()
        img = img.to(torch.device('cpu'))
        img = img.permute(1, 2, 0).numpy()
        img = img + np.array(mean, dtype=np.float32)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (height, width))
        return img

    # return origin BGR image or with HSV change 
    @staticmethod
    def randomHSV(img, item='v'):
        if random.random() < 0.5:
            adjustRange = [0.5, 1.5]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if item == 'h':
                h = np.clip(h * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            elif item == 's':
                s = np.clip(s * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            elif item == 'v':
                v = np.clip(v * random.choice(adjustRange), 0, 255).astype(hsv.dtype)
            img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        return img

    # return origin BGR image or blurred image 
    @staticmethod
    def randomBlur(img):
        if random.random() < 0.5:
            img = cv2.blur(img, (5,5))
        return img

    # return origin image or with darken and noise  
    @staticmethod
    def randomNoise(img, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            img = img * alpha + random.randrange(-delta, delta)
            img = img.clip(min=0, max=255).astype(np.uint8)
        return img

    # return origin image or fliped left and right (mirror image)
    @staticmethod
    def randomFlip(img, boxes):
        if random.random() < 0.5:
            img_lr = np.fliplr(img).copy()
            _, w, _ = img.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return img_lr, boxes
        return img, boxes

    # return origin BGR image or with width re scaled 
    @staticmethod
    def randomScale(img, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height, width, _ = img.shape
            img = cv2.resize(img, (int(width*scale), height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return img, boxes
        return img, boxes

    # return origin image or cropped image
    @staticmethod
    def randomCrop(img, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2.0
            height, width, _ = img.shape
            h = random.uniform(0.6*height, height)
            w = random.uniform(0.6*width, width)
            x = random.uniform(0, width-w)
            y = random.uniform(0, height-h)
            x, y, h, w = int(x), int(y), int(h), int(w)
            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0] > 0) & (center[:,0] < w)
            mask2 = (center[:,1] > 0) & (center[:,1] < h)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return img,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)
            boxes_in = boxes_in - box_shift
            # to make boxes_in really inside this image
            boxes_in[:,0] = boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2] = boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1] = boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3] = boxes_in[:,3].clamp_(min=0,max=h)
            labels_in = labels[mask.view(-1)]
            img_cropped = img[y:y+h, x:x+w, :]
            return img_cropped, boxes_in, labels_in
        return img, boxes, labels

    # return origin BGR image or shifted (uncovered area are as default)
    @staticmethod
    def randomShift(img, boxes, labels):
        if random.random() < 0.5:
            default = [104,117,123] # default color            
            img_shifted = np.zeros(img.shape, dtype=img.dtype)
            img_shifted[:,:,:] = default
            height, width, _ = img.shape
            shift_x = int(random.uniform(-width*0.2, width*0.2))
            shift_y = int(random.uniform(-height*0.2, height*0.2))
            # shift image ------------------------------
            if shift_x >= 0 and shift_y >= 0:
                img_shifted[shift_y:, shift_x:, :] = img[:height-shift_y, :width-shift_x, :]
            elif shift_x >= 0 and shift_y < 0:
                img_shifted[:height+shift_y, shift_x:, :] = img[-shift_y:, :width-shift_x, :]
            elif shift_x < 0 and shift_y >= 0:
                img_shifted[shift_y:, :width+shift_x, :] = img[:height-shift_y, -shift_x:, :]
            elif shift_x < 0 and shift_y < 0:
                img_shifted[:height+shift_y, :width+shift_x, :] = img[-shift_y:, -shift_x:, :]
            # shift box ------------------------------
            center = (boxes[:, 2:] + boxes[:, :2])/2
            shift_xy = torch.FloatTensor([[shift_x, shift_y]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] > 0) & (center[:,0] < width)
            mask2 = (center[:,1] > 0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            # question: if center is in but box is out, what to do, follow modified may fix this ??????
            inside = [max([shift_x, 0]), max([shift_y, 0]), min([width - shift_x]), min([height - shift_y])]
            boxes_in[:,0] = boxes_in[:,0].clamp_(min=inside[0],max=inside[2])
            boxes_in[:,2] = boxes_in[:,2].clamp_(min=inside[0],max=inside[2])
            boxes_in[:,1] = boxes_in[:,1].clamp_(min=inside[1],max=inside[3])
            boxes_in[:,3] = boxes_in[:,3].clamp_(min=inside[1],max=inside[3])
            if len(boxes_in) != 0:
                box_shift = torch.FloatTensor([[shift_x, shift_y, shift_x, shift_y]]).expand_as(boxes_in)
                boxes_in = boxes_in + box_shift
                labels_in = labels[mask.view(-1)]
                return img_shifted, boxes_in, labels_in
        return img, boxes, labels


# used as a dataset for data.Dataloader
class yoloDataset(data.Dataset):
    def __init__(self, root, list_file, train):
        self.root = str(root) + ('/' if str(root)[-1] != '/' else '') 
        self.train = train
        self.loadBoxLabel = targetGenerator.loadBoxLabel
        self.genTarget = targetGenerator.genTarget
        self.imageTensor = imageTransform.imageTensor 
        self.randomFlip = imageTransform.randomFlip
        self.randomScale = imageTransform.randomScale
        self.randomBlur = imageTransform.randomBlur
        self.randomHSV = imageTransform.randomHSV
        self.randomNoise = imageTransform.randomNoise
        self.randomShift = imageTransform.randomShift
        self.randomCrop = imageTransform.randomCrop
        self.filenames, self.labels, self.boxes = self.loadBoxLabel(list_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = cv2.imread(self.root + filename)
        boxes = torch.Tensor(self.boxes[idx]).clone()
        labels = torch.Tensor(self.labels[idx]).clone()

        if self.train:
            img, boxes = self.randomFlip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.randomHSV(img, 'v')
            img = self.randomHSV(img, 'h')
            img = self.randomHSV(img, 's')
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        target = self.genTarget(h, w, boxes, labels)
        img = self.imageTensor(img)
        return img, target



if __name__ == '__main__':

    test_xml = False
    annotation_folder = '/home/LiuYue/mlsFolder/VOC/VOC2007test2012train/VOCdevkit/VOC2007/Annotations/'
    out_file = 'tmp_test_voc2007test.txt'

    test_loss = False
    loss_save = 'tmp_loss_save.pkl'

    test_dataset = False
    image_folder = './allimgs'
    list_file = ['voc2007.txt', 'voc2007test.txt']

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hl:a:o:d:f:", ['help', 'loss=', 'anno=', 'out=', 'dir=', 'file='])
    except getopt.GetoptError:
        print('usage: python *.py (loss) [-l | --loss filename]')
        print('usage: python *.py (xml) [-a | --anno filename] [-o | --out filename]')
        print('usage: python *.py (dataset) [-d | --dir foldername] [-f | --file filenames, as \'1.txt+2.txt\']')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: python *.py (loss) [-l | --loss filename]')
            print('usage: python *.py (xml) [-a | --anno filename] [-o | --out filename]')
            print('usage: python *.py (dataset) [-d | --dir foldername] [-f | --file filenames, as \'1.txt+2.txt\']')
            sys.exit(2)
        elif opt in ('-l', '--loss'):
            test_loss = True
            loss_save = arg
        elif opt in ('-a', '--anno'):
            test_xml = True
            annotation_folder = arg
        elif opt in ('-o', '--out'):
            test_xml = True
            out_file = arg
        elif opt in ('-d', '--dir'):
            test_dataset = True
            image_folder = arg
        elif opt in ('-f', '--file'):
            test_dataset = True
            list_file = arg.split('+')

    # print(list_file)

    if test_xml:
        targetGenerator.VOCXml2Txt(annotations=annotation_folder, out_file=out_file)

    if test_loss:
        with open(loss_save, "rb") as f:
            data = pickle.load(f) 
        loss_test = []
        loss_train = []
        for i in range(len(data)):
            for j in range(len(data[i]['loss_test'])):
                loss_test.append(float(data[i]['loss_test'][j]))
            for j in range(len(data[i]['loss_train'])):
                loss_train.append(float(data[i]['loss_train'][j]))

        imageShow.curveShow(list(range(len(loss_test))), loss_test, title='loss test', xlabel='epoch', ylabel='loss')
        time.sleep(20)
        imageShow.curveShow(list(range(len(loss_train))), loss_train, title='loss train (923 iters / epoch)', xlabel='iter', ylabel='loss')

    if test_dataset:
        dataset = yoloDataset(root=image_folder, list_file=list_file, train=True)
        for i in range(len(dataset)):
            img, target = dataset[int(len(dataset) * random.random())]
            # print('image size: {}, target size: {}.'.format(img.size(), target.size()))
            _, h, w = img.size()
            img = imageTransform.imageRestore(img, h, w)
            probs, labels, boxes = targetGenerator.parseTarget(target, h, w)
            # print(probs, labels, boxes)
            imageShow.boxImageShow(img, probs, labels, boxes, resize=False)
            time.sleep(3)

