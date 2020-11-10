import torch
import torch.nn as nn


class yoloLoss(nn.Module):
    def __init__(self, grid_num, box_num, label_num, coord_rate, noobj_rate):
        super(yoloLoss, self).__init__() # init nn.Module
        self.box_num = box_num
        self.label_num = label_num
        self.all_box_num = 5 * box_num
        self.target_num = 5 * box_num + label_num
        self.cell_size = 1.0 / grid_num
        self.coord_rate = coord_rate # rate for coordinate loss
        self.noobj_rate = noobj_rate # rate for no object loss 

    @staticmethod
    def calcIOUs(box_infos1, box_infos2, cell_size):
        # box_infos: Nx4 tensor, Mx4 tensor, box_infos[i] = [dx, dy, w, h]
        # boxes1: Nx4 tensor, Mx4 tensor, boxes2[i] = [x1, y1, x2, y2]
        # attention: cx, cy can be relative, but do not have to get the real center!!!
        # return ious: NxM tensor
        # do not +1.0 in x or y since x or y have been normalized 
        # WARNING: do not use a = a and is modified, that would make torch.autograd confused 

        boxes1 = box_infos1 * 1.0
        boxes2 = box_infos2 * 1.0
        centers1 = cell_size * box_infos1[:,0:2]
        centers2 = cell_size * box_infos2[:,0:2]
        sizes1 = box_infos1[:,2:4] / 2.0
        sizes2 = box_infos2[:,2:4] / 2.0
        boxes1[:,0:2] = centers1 - sizes1
        boxes1[:,2:4] = centers1 + sizes1
        boxes2[:,0:2] = centers2 - sizes2
        boxes2[:,2:4] = centers2 + sizes2

        N = boxes1.size(0)
        M = boxes2.size(0)
        # calc areas
        areas1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        areas2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
        areas1 = areas1.unsqueeze(1).expand(N, M) # N -> Nx1 -> NxM
        areas2 = areas2.unsqueeze(0).expand(N, M) # M -> Mx1 -> NxM

        # calc intersections, as left_tops[i,j,:] = max(boxes1[i,:], boxes2[j,:])
        left_top_boxes1 = boxes1[:,0:2].unsqueeze(1).expand(N,M,2) # Nx2 -> Nx1x2 -> NxMx2
        left_top_boxes2 = boxes2[:,0:2].unsqueeze(0).expand(N,M,2) # Mx2 -> 1xMx2 -> NxMx2
        right_bottom_boxes1 = boxes1[:,2:4].unsqueeze(1).expand(N,M,2) # Nx2 -> Nx1x2 -> NxMx2
        right_bottom_boxes2 = boxes2[:,2:4].unsqueeze(0).expand(N,M,2) # Mx2 -> 1xMx2 -> NxMx2        
        left_tops = left_top_boxes1.max(left_top_boxes2)
        right_bottoms = right_bottom_boxes1.min(right_bottom_boxes2)
        width_heights = (right_bottoms - left_tops).clamp(min=0)
        intersections = width_heights[:,:,0] * width_heights[:,:,1]
        # unions = (areas1 + areas2 - intersections).clamp(min=0.000000000001) # avoid 0
        ious = intersections / (areas1 + areas2 - intersections)

        return ious

    def forward(self, preds, truths):
        # torch.autograd.set_detect_anomaly(True)
        # maybe the key is detach, maybe x * 0 is also calculated in backward, that is just maybe.
        # remember only preds need to calc grad, no mask or truths

        cell_size = self.cell_size
        target_num = self.target_num
        all_box_num = self.all_box_num
        coord_rate = self.coord_rate 
        noobj_rate = self.noobj_rate
        calcIOUs = self.calcIOUs
        batch_size = preds.size(0)

        mask_truths_obj = (truths[:,:,:,4] > 0).unsqueeze(-1).expand_as(truths)
        preds_obj = preds[mask_truths_obj].view(-1, target_num)
        truths_obj = truths[mask_truths_obj].view(-1, target_num)
        preds_no_obj = preds[mask_truths_obj.logical_not()].view(-1, target_num)
        preds_obj_box = preds_obj[:, 0:all_box_num]
        truths_obj_box = truths_obj[:, 0:all_box_num]

        preds_no_obj_conf = preds_no_obj[:, list(range(4, all_box_num, 5))]
        truths_no_obj_conf = (preds_no_obj_conf * 0.0).detach()

        truths_obj_conf = (truths_obj[:, 0] * 0.0) # truth confidence = prob * iou = iou 
        mask_preds_obj_response = (truths_obj_box * 0.0).bool() # as indication_obj_reponse
        # get coordinates for each grids to calc ious
        # as in truths, one grid has only one box (boxes share same information)
        for grid in range(truths_obj.size(0)):            
            preds_in_grid_coor = preds_obj_box[grid,:].view(-1,5)[:,0:4] # size: box_num * 4
            truths_in_grid_coor = truths_obj_box[grid,0:4].view(1,4) # size:1 * 4
            ious = calcIOUs(preds_in_grid_coor, truths_in_grid_coor, cell_size=cell_size).view(-1)
            max_iou, tmp_ind = ious.max(0) # if not consider confs, '#' below
            # if ious are the same, we tend to choose the higher confidence one to aviod random output loss
            tmp_ind = (ious == max_iou).nonzero().view(-1)
            if tmp_ind.size(0) > 1:
                preds_in_grid_conf = preds_obj_box[grid, tmp_ind + 4].view(-1)
                tmp_ind = preds_in_grid_conf.max(0)[1]
            mask_preds_obj_response[grid, tmp_ind*5:tmp_ind*5+5] = True
            truths_obj_conf[grid] = max_iou.item()

        preds_obj_response = preds_obj_box[mask_preds_obj_response].view(-1,5) # size: num * 5 
        truths_obj_response = truths_obj_box[:,0:5] # all the same, can use mask too
        preds_obj_no_response_conf = preds_obj_box[mask_preds_obj_response.logical_not()].view(-1,5)[:, 4]
        truths_obj_no_response_conf = (preds_obj_no_response_conf * 0.0).detach()

        center_loss = nn.functional.mse_loss(preds_obj_response[:,0:2], truths_obj_response[:,0:2], reduction='sum')
        size_loss = nn.functional.mse_loss(preds_obj_response[:,2:4].sqrt(), truths_obj_response[:,2:4].sqrt(), reduction='sum')
        conf_obj_response_loss = nn.functional.mse_loss(preds_obj_response[:,4], truths_obj_conf, reduction='sum')
        # indication_noobj probabaly means 'not indication_obj_reponse', 
        # thus contains no object and have object but with no responsible 
        conf_obj_no_response_loss = nn.functional.mse_loss(preds_obj_no_response_conf, truths_obj_no_response_conf, reduction='sum')
        conf_noobj_loss = nn.functional.mse_loss(preds_no_obj_conf, truths_no_obj_conf, reduction='sum')
        label_loss = nn.functional.mse_loss(preds_obj[:,all_box_num:], truths_obj[:,all_box_num:], reduction='sum')

        return (coord_rate * (center_loss + size_loss) + conf_obj_response_loss + noobj_rate *  conf_obj_no_response_loss + noobj_rate * conf_noobj_loss + label_loss) / batch_size # ori loss

        # return (coord_rate * (center_loss + size_loss) + 2 * conf_obj_response_loss + conf_obj_no_response_loss + noobj_rate * conf_noobj_loss + label_loss) / batch_size # origin author's loss

