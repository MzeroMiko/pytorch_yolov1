import torch
import numpy as np


# function class parsing target
class targetParser():
    
    def __init__(self, box_num=2, image_width=1.0, image_height=1.0, threshold=0.1):
        self.box_num = box_num
        self.threshold = threshold
        self.image_width = image_width
        self.image_height = image_height

    # non maximum suppression, return list keep whith input np.array boxes
    @staticmethod
    def nms(boxes, threshold=0.5):
        # boxes: np.array([[x1,y1,x2,y2, probs],...])
        keep = []
        order = np.argsort(boxes[:,4])[::-1]
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
            unions = np.maximum(areas[i] + areas[order] - intersections, np.finfo(np.float64).eps)
            ious = intersections / unions
            order = order[np.where(ious <= threshold)[0]]
        
        return keep

    # return np.array label_boxes([x1,y1,x2,y2,prob,label]) with input np.array target, use nms
    def parse(self, target):
        label_boxes = [] # [[x1,y1,x2,y2,prob,label],...]
        box_num = self.box_num
        threshold = self.threshold
        all_box_num = 5 * box_num
        height, width, target_num = target.shape
        grid_num = height * width

        target = target.reshape(grid_num, target_num)
        target_class = np.transpose(np.tile(target[:, all_box_num:],(box_num, 1, 1)), (1, 0, 2))
        target_labels = np.argmax(target_class, axis=2)
        target_label_probs = np.max(target_class, axis=2)

        linspace_width = np.transpose(np.tile(np.arange(0, width), (box_num, height,1)), (1, 2, 0))
        linspace_height = np.transpose(np.tile(np.arange(0, height), (box_num, width,1)), (2, 1, 0))
        linspace_width = linspace_width.reshape(grid_num, box_num)
        linspace_height = linspace_height.reshape(grid_num, box_num)

        target_tboxes = target[:, 0:all_box_num].reshape(grid_num, box_num, 5)
        target_centers = target_tboxes[:, :, 0:2] * 1.0
        target_centers[:, :, 0] = (target_tboxes[:, :, 0] + linspace_width) / width
        target_centers[:, :, 1] = (target_tboxes[:, :, 1] + linspace_height) / height
        target_half_sizes = target_tboxes[:, :, 2:4] / 2.0
        target_label_boxes = np.zeros((grid_num, box_num, 6))
        target_label_boxes[:,:,0:2] = np.maximum(target_centers - target_half_sizes, 0.0)
        target_label_boxes[:,:,2:4] = np.minimum(target_centers + target_half_sizes, 1.0)
        target_label_boxes[:,:,4] = target_tboxes[:,:,4] * target_label_probs
        target_label_boxes[:,:,5] = target_labels
        target_label_boxes[:,:,0] = target_label_boxes[:,:,0] * self.image_width
        target_label_boxes[:,:,1] = target_label_boxes[:,:,1] * self.image_height
        target_label_boxes[:,:,2] = target_label_boxes[:,:,2] * self.image_width
        target_label_boxes[:,:,3] = target_label_boxes[:,:,3] * self.image_height
        
        # label_boxes = target_label_boxes.reshape(-1, 6)
        # label_boxes = label_boxes[np.where(label_boxes[:,4] > threshold)[0], :]
        # label_boxes = label_boxes[self.nms(label_boxes), :] if label_boxes.shape[0] != 0 else label_boxes
        # return label_boxes

        for i in range(grid_num):
            unique_boxes = np.array(list(set([tuple(box) for box in list(target_label_boxes[i, :, :])])))
            chosen_boxes = unique_boxes[np.where(unique_boxes[:,4] > threshold)[0], :]
            label_boxes.extend(chosen_boxes.tolist())
                        
        # non maximum suppression 
        label_boxes = np.array(label_boxes)
        label_boxes = label_boxes[self.nms(label_boxes), :] if label_boxes.shape[0] != 0 else label_boxes
        return label_boxes


# yolo v1 loss function
class yoloLoss(torch.nn.Module):
    
    def __init__(self, box_num=2, coord_rate=5.0, noobj_rate=0.5):
        super(yoloLoss, self).__init__() # init nn.Module
        self.box_num = box_num
        self.coord_rate = coord_rate
        self.noobj_rate = noobj_rate
        self.mse_loss = torch.nn.functional.mse_loss

    @staticmethod
    def calcIOUs(boxes1, boxes2):
        # return ious: NxM tensor, input tensor Nx4 and Mx4 boxes = [[x1,x2,y1,y2],...]
        # do not +1.0 in x or y since x or y have been normalized 
        # little trick: torch.Size(4,1) + torch.Size(1,5) = torch.Size(4,5) that means broadcast
        # and numpy would do the same

        areas1 = ((boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])).view(-1, 1)
        areas2 = ((boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])).view(1, -1)
        xmax = boxes1[:,2].view(-1,1).min(boxes2[:,2].view(1,-1))
        xmin = boxes1[:,0].view(-1,1).max(boxes2[:,0].view(1,-1))
        ymax = boxes1[:,3].view(-1,1).min(boxes2[:,3].view(1,-1))
        ymin = boxes1[:,1].view(-1,1).max(boxes2[:,1].view(1,-1))
        intersections = (xmax - xmin).clamp(min=0.0) * (ymax - ymin).clamp(min=0.0)
        unions = (areas1 + areas2 - intersections).clamp(min=0.000000000001) # avoid 0
        ious = intersections / unions
        return ious
        
    def forward(self, output, truths, iteration):
        # tensor output: batch_size  * height * width * (5*box_num * class_num); 
        # tensor truths: batch_size * obj_num * 5 [[[x1,y1,x2,y2,label],...]]
        # note that obj_num may differ in every batch, and with padding x=0 in it to make it a tensor
        # tboxes:[tx,ty,w,h,prob/label], cboxes: [cx,cy,w,h,prob/label], boxes: [x1,y1,x2,y2,prob/label]
        # needed for autograd (do not detach!): pred_obj_class, pred_obj_tboxes, pred_noobj_probs, pred_tboxes
        batch_size, height, width, target_num  = output.size()
        all_box_num = 5 * self.box_num
        class_num = int(target_num - all_box_num)
        truths_obj_num = truths.size(1)

        preds = output.contiguous().view(batch_size, height * width, target_num)
        pred_class = preds[:,:,all_box_num:].contiguous()
        pred_tboxes = preds[:,:,0:all_box_num].view(batch_size, height * width, self.box_num, 5).contiguous()

        with torch.no_grad():
            linspace_height = torch.arange(0, height, device=preds.device).repeat(batch_size, width, 1).permute(0, 2, 1).contiguous().view(batch_size, -1)
            linspace_width = torch.arange(0, width, device=preds.device).repeat(batch_size, height, 1).contiguous().view(batch_size, -1)

            pred_cboxes = pred_tboxes.clone()
            pred_cboxes[:,:,:,0] = (pred_tboxes[:,:,:,0] + linspace_width.repeat(self.box_num, 1, 1).permute(1, 2, 0)) /width
            pred_cboxes[:,:,:,1] = (pred_tboxes[:,:,:,1] + linspace_height.repeat(self.box_num, 1, 1).permute(1, 2, 0)) /height
            pred_boxes = pred_cboxes.clone()
            pred_boxes[:,:,:,0:2] = pred_cboxes[:,:,:,0:2] - pred_cboxes[:,:,:,2:4] / 2.0
            pred_boxes[:,:,:,2:4] = pred_cboxes[:,:,:,0:2] + pred_cboxes[:,:,:,2:4] / 2.0

            truth_boxes = truths.to(preds.device)
            truth_cboxes = truth_boxes.clone()
            truth_cboxes[:,:,0:2] = (truth_boxes[:,:,0:2] + truth_boxes[:,:,2:4]) / 2.0
            truth_cboxes[:,:,2:4] = (truth_boxes[:,:,2:4] - truth_boxes[:,:,0:2])
            truth_delta_x = truth_cboxes[:,:,0] * width
            truth_delta_y = truth_cboxes[:,:,1] * height
            truth_grid_x = truth_delta_x.ceil() - 1
            truth_grid_y = truth_delta_y.ceil() - 1
            truth_delta_x = truth_delta_x - truth_grid_x
            truth_delta_y = truth_delta_y - truth_grid_y
            truth_grid_pos = truth_grid_y * width + truth_grid_x

            # find the best_in_grid_box for each grid truth; get mask and related truths
            # obj means that box is responsible for one truth box, else noobj
            mask_response = (preds * 0.0).bool().detach()
            truth_response = (preds * 0.0).float().detach()
            for b in range(batch_size):
                for k in range(truths_obj_num):
                    truth_grid_boxes = truth_boxes[b, k, 0:4].view(-1,4)
                    if truth_grid_boxes.sum() == 0: continue # if truth box has no use  
                    position = int(truth_grid_pos[b, k])
                    pred_grid_boxes = pred_boxes[b, position, :, 0:4]
                    iou_tensor = self.calcIOUs(truth_grid_boxes, pred_grid_boxes)
                    max_ious, inds = iou_tensor.max(1) # get the best match pred_box for truth
                    max_iou, ind = max_ious[0], int(inds[0])
                    if max_iou == 0: continue # no use wasting time on this
                    # if that position has been filled
                    if mask_response[b, position, all_box_num] == True:
                        prev_iou = truth_response[b, position, list(range(4, all_box_num, 5))].sum() 
                        if prev_iou >= max_iou: # reverse previous data
                            continue
                        else: # clear previous data
                            mask_response[b, position, 0:all_box_num] = False
                            truth_response[b, position, :] = 0.0   
                    box_ind = 5 * ind
                    mask_response[b, position, box_ind:box_ind+5] = True
                    mask_response[b, position, all_box_num:] = True
                    truth_response[b, position, box_ind] = truth_delta_x[b, k]
                    truth_response[b, position, box_ind+1] = truth_delta_y[b, k]
                    truth_response[b, position, box_ind+2:box_ind+4] = truth_cboxes[b, k, 2:4]
                    truth_response[b, position, box_ind+4] = max_iou
                    truth_response[b, position, int(all_box_num + truth_cboxes[b,k,4])] = 1.0

            mask_tbox_response = mask_response[:,:,0:all_box_num].contiguous().view(-1,5)
            mask_class_response = mask_response[:,:,all_box_num:].contiguous().view(-1,class_num)
            truth_tbox_response = truth_response[:,:,0:all_box_num].contiguous().view(-1,5)
            truth_class_response = truth_response[:,:,all_box_num:].contiguous().view(-1, class_num)
        
        pred_obj_tboxes = pred_tboxes.view(-1,5)[mask_tbox_response].view(-1,5)
        truth_obj_tboxes = truth_tbox_response[mask_tbox_response].view(-1,5)
        pred_obj_class = pred_class.view(-1, class_num)[mask_class_response].view(-1,class_num)
        truth_obj_class = truth_class_response[mask_class_response].view(-1,class_num)
        pred_noobj_probs = pred_tboxes.view(-1,5)[mask_tbox_response.logical_not()].view(-1,5)[:,4]
        truth_noobj_probs = (pred_noobj_probs * 0.0).detach()

        loss_center = self.coord_rate * self.mse_loss(pred_obj_tboxes[:,0:2], truth_obj_tboxes[:,0:2], reduction='sum')
        loss_size = self.coord_rate * self.mse_loss(pred_obj_tboxes[:,2:4].sqrt(), truth_obj_tboxes[:,2:4].sqrt(), reduction='sum')
        loss_conf_obj = self.mse_loss(pred_obj_tboxes[:,4], truth_obj_tboxes[:,4], reduction='sum')
        loss_conf_noobj = self.noobj_rate * self.mse_loss(pred_noobj_probs, truth_noobj_probs, reduction='sum')
        loss_class = self.mse_loss(pred_obj_class, truth_obj_class, reduction='sum')
        loss_total = loss_center + loss_size + loss_conf_obj + loss_conf_noobj + loss_class

        # print(loss_center, loss_size, loss_conf_obj, loss_conf_noobj,loss_class)
        return loss_total / batch_size


if __name__ == "__main__":
    batch_size = 24
    preds = torch.rand((batch_size,14,14,30))
    preds.requires_grad = True
    preds.retain_graph = True
    tmp = torch.rand((batch_size, 4, 4)) * 0.8 + 0.1 
    truths = torch.zeros((batch_size, 4, 5))
    truths[:,:,0] = torch.min(tmp[:,:,0:2], axis=2)[0]
    truths[:,:,1] = torch.min(tmp[:,:,2:4], axis=2)[0]
    truths[:,:,2] = torch.max(tmp[:,:,0:2], axis=2)[0]
    truths[:,:,3] = torch.max(tmp[:,:,2:4], axis=2)[0]
    truths[:,:,4] = 13
    # print(truths)
    citer = yoloLoss()
    loss = citer(preds, truths, 0)
    loss.backward()
    print(loss, preds.grad.sum())
    label_boxes = targetParser().parse(preds[0,:,:,:].detach().numpy())
    print(label_boxes)
    import time
    start = time.time() # s
    for b in range(128):
        loss = citer(preds, truths, 0)
        loss.backward()
    print((time.time()-start)/128)
    

