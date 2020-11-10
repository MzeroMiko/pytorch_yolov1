import torch
import numpy as np

# function class parsing target
class targetParser():

    def __init__(self, anchors=None, image_width=1.0, image_height=1.0, threshold=0.1):
        self.anchors = anchors if type(anchors) != type(None) else np.array([
            (1.3221/13, 1.73145/13), (3.19275/13, 4.00944/13),
            (5.05587/13, 8.09892/13), (9.47112/13, 4.84053/13), 
            (11.2364/13, 10.0071/13)]) 
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
        anchors = self.anchors
        threshold = self.threshold        
        anchor_num = anchors.shape[0]
        height, width, _ = target.shape
        grid_num = height * width

        sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))

        target = target.reshape(grid_num, anchor_num, -1)
        target_class = target[:, :, 5:]
        target_labels = np.argmax(target_class, axis=2)
        target_label_probs = np.max(target_class, axis=2)

        linspace_width = np.transpose(np.tile(np.arange(0, width), (anchor_num, height,1)), (1, 2, 0))
        linspace_height = np.transpose(np.tile(np.arange(0, height), (anchor_num, width,1)), (2, 1, 0))
        linspace_width = linspace_width.reshape(grid_num, anchor_num)
        linspace_height = linspace_height.reshape(grid_num, anchor_num)

        target_tboxes = target[:,:,0:5]
        target_centers = target_tboxes[:,:,0:2] * 1.0
        target_centers[:,:,0] = (sigmoid(target_tboxes[:,:,0]) + linspace_width) / width
        target_centers[:,:,1] = (sigmoid(target_tboxes[:,:,1]) + linspace_height) / height
        target_half_sizes = np.exp(target_tboxes[:,:,2:4]) * np.tile(anchors, (grid_num, 1, 1)) / 2.0
        target_label_boxes = np.zeros((grid_num, anchor_num, 6))
        target_label_boxes[:,:,0:2] = np.maximum(target_centers - target_half_sizes, 0.0)
        target_label_boxes[:,:,2:4] = np.minimum(target_centers + target_half_sizes, 1.0)
        target_label_boxes[:,:,4] = sigmoid(target_tboxes[:,:,4]) * target_label_probs
        target_label_boxes[:,:,5] = target_labels
        target_label_boxes[:,:,0] = target_label_boxes[:,:,0] * self.image_width
        target_label_boxes[:,:,1] = target_label_boxes[:,:,1] * self.image_height
        target_label_boxes[:,:,2] = target_label_boxes[:,:,2] * self.image_width
        target_label_boxes[:,:,3] = target_label_boxes[:,:,3] * self.image_height

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
    
    def __init__(self, 
        anchors = torch.Tensor([
            (1.3221/13, 1.73145/13), 
            (3.19275/13, 4.00944/13), 
            (5.05587/13, 8.09892/13), 
            (9.47112/13, 4.84053/13), 
            (11.2364/13, 10.0071/13)]), 
        thresh=0.6, 
        prior_iter=12800, 
        prior_rate=1.0, 
        coord_rate=1.0, 
        noobj_rate=1.0, 
        object_rate=5.0, 
        class_rate=1.0
        ):

        super(yoloLoss, self).__init__()
        # tensor anchors: [[anc_w, anc_h],...]
        self.thresh = thresh
        self.anchors = torch.Tensor(anchors)
        
        self.prior_iter = prior_iter
        self.prior_rate = prior_rate
        self.coord_rate = coord_rate
        self.noobj_rate = noobj_rate
        self.class_rate = class_rate
        self.object_rate = object_rate
        self.mse_loss = torch.nn.functional.mse_loss
    
    @staticmethod
    def calcIOUs(boxes1, boxes2):
        # return ious: NxM tensor, input tensor Nx4 and Mx4 boxes = [[x1,x2,y1,y2],...]
        # do not +1.0 in x or y since x or y have been normalized 
        # WARNING: do not use a = a and is modified, that would make torch.autograd confused 
        # little trick: torch.Size(4,1) + torch.Size(1,5) = torch.Size(4,5) that means broadcast
        # and numpy would do the same

        areas1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        areas2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
        xmax = boxes1[:,2].view(-1,1).min(boxes2[:,2].view(1,-1))
        xmin = boxes1[:,0].view(-1,1).max(boxes2[:,0].view(1,-1))
        ymax = boxes1[:,3].view(-1,1).min(boxes2[:,3].view(1,-1))
        ymin = boxes1[:,1].view(-1,1).max(boxes2[:,1].view(1,-1))
        intersections = (xmax - xmin).clamp(min=0.0) * (ymax - ymin).clamp(min=0.0)
        unions = (areas1 + areas2.t() - intersections).clamp(min=0.000000000001) # avoid 0
        ious = intersections / unions

        return ious

    def forward(self, output, truths, iteration):
        # height = grid_num in height, width = grid_num in width 
        # tensor output: batch_size * (anchor_num * (5 + class_num)) * height * width; 
        # tensor truths: batch_size * obj_num * 5 [[[x1,y1,x2,y2,label],...]]
        # pred_cboxes: [cx,cy,w,h,prob], pred_boxes: [x1,y1,x2,y2,prob]
        # truth_cboxes: [cx,cy,w,h,label], truth_boxes: [x1,y1,x2,y2,label]
        # needed for autograd (do not detach!): pred_obj_class, pred_obj_cboxes, pred_noobj_probs, pred_cboxes
        # Attention: in the net work, data has been sigmoided !!!

        batch_size, height, width, target_num = output.size()
        anchor_num = self.anchors.size(0)
        class_num = int(target_num / anchor_num - 5)
        truths_obj_num = truths.size(1)
            
        reverse_sigmoid = lambda y: -1 * (1 / y - 1).log() 

        anchors = self.anchors.to(output.device).repeat(batch_size, height * width, 1, 1)
        preds = output.contiguous().view(batch_size, height * width, anchor_num, -1)
        pred_class = preds[:,:,:,5:]
        pred_tboxes = preds[:,:,:,0:5]

        linspace_height = torch.arange(0, height, device=preds.device).repeat(batch_size, width, 1).permute(0, 2, 1).contiguous().view(batch_size, -1)
        linspace_width = torch.arange(0, width, device=preds.device).repeat(batch_size, height, 1).contiguous().view(batch_size, -1)

        # bx = (sigmoid(tx) + gw) / width; bw = pw * exp(tw)
        pred_cboxes = (pred_tboxes * 1.0)
        pred_cboxes[:,:,:,0] = (pred_tboxes[:,:,:,0].sigmoid() + linspace_width.repeat(anchor_num, 1, 1).permute(1, 2, 0)) / width
        pred_cboxes[:,:,:,1] = (pred_tboxes[:,:,:,1].sigmoid() + linspace_height.repeat(anchor_num, 1, 1).permute(1, 2, 0)) / height
        pred_cboxes[:,:,:,2:4] = pred_tboxes[:,:,:,2:4].exp() * anchors
        pred_cboxes[:,:,:,4] = pred_tboxes[:,:,:,4].sigmoid()
        pred_boxes = (pred_cboxes * 1.0)
        pred_boxes[:,:,:,0:2] = pred_cboxes[:,:,:,0:2] - pred_cboxes[:,:,:,2:4] / 2.0
        pred_boxes[:,:,:,2:4] = pred_cboxes[:,:,:,0:2] + pred_cboxes[:,:,:,2:4] / 2.0 

        with torch.no_grad():
            truth_boxes = truths.to(preds.device)
            truth_cboxes = truth_boxes * 1.0
            truth_cboxes[:,:,0:2] = (truth_boxes[:,:,0:2] + truth_boxes[:,:,2:4]) / 2.0
            truth_cboxes[:,:,2:4] = (truth_boxes[:,:,2:4] - truth_boxes[:,:,0:2])
            truth_delta_x = truth_cboxes[:,:,0] * width
            truth_delta_y = truth_cboxes[:,:,1] * height
            truth_grid_x = truth_delta_x.ceil() - 1
            truth_grid_y = truth_delta_y.ceil() - 1
            truth_delta_x = reverse_sigmoid(truth_delta_x - truth_grid_x)
            truth_delta_y = reverse_sigmoid(truth_delta_y - truth_grid_y)

            fake_truth_boxes = truth_boxes * 1.0
            fake_truth_boxes[:,:,0:2] = 0.0 - truth_cboxes[:,:,2:4] / 2.0
            fake_truth_boxes[:,:,2:4] = 0.0 + truth_cboxes[:,:,2:4] / 2.0
            fake_anchor_boxes = pred_boxes[0,0,:,:] * 1.0
            fake_anchor_boxes[:,0:2] = 0.0 - anchors[0,0,:,:] / 2.0
            fake_anchor_boxes[:,2:4] = 0.0 + anchors[0,0,:,:] / 2.0

            # find the best_in_grid_box for each grid truth; get mask and related truths
            # obj means that the box (anchor) is responsible for one truth box, else noobj
            # only noobj_response would be take into account when calc loss_noobj
            mask_response = (preds * 0.0).bool().detach()
            truth_response = (preds * 0.0).float().detach()
            truth_response_size = (truth_response[:,:,:,2:4] * 0.0).detach()
            mask_noobj_response = (pred_boxes[:,:,:,4] * 0.0).bool().detach()
            for b in range(batch_size):
                pred_batch_boxes = pred_boxes[b,:,:,0:4].view(-1,4)
                truth_batch_boxes = truth_boxes[b,:,0:4].view(-1,4)
                pred_truth_noobj_iou_tensor = self.calcIOUs(pred_batch_boxes, truth_batch_boxes)
                max_ious, _ = pred_truth_noobj_iou_tensor.max(1)
                mask_noobj_response[b, :, :] = (max_ious < self.thresh).view(height * width, anchor_num)

                for k in range(truths_obj_num):
                    position = int(truth_grid_y[b, k] * width + truth_grid_x[b, k])
                    fake_truth_grid_boxes = fake_truth_boxes[b, k, 0:4].view(-1,4)
                    fake_anchor_grid_boxes = fake_anchor_boxes[:, 0:4].view(-1,4)
                    truth_anchor_iou_tensor = self.calcIOUs(fake_truth_grid_boxes, fake_anchor_grid_boxes)
                    max_ious, inds = truth_anchor_iou_tensor.max(1) # get the best match anchor for truth
                    max_iou, ind = max_ious[0], int(inds[0])
                    if max_iou == 0: continue # no use wasting time on this
                    label_ind = int(5 + truth_cboxes[b, k, 4])
                    # if that position has been filled
                    if mask_response[b, position, ind, 0] == True:
                        prev_iou = truth_response[b, position, ind, 4] 
                        if prev_iou >= max_iou: # reverse previous data
                            continue
                        else: # clear previous data
                            # mask_response[b, position, ind, :] = False
                            truth_response[b, position, ind, :] = 0.0   
                    mask_response[b, position, ind, :] = True
                    truth_response[b, position, ind, 0] = truth_delta_x[b, k]
                    truth_response[b, position, ind, 1] = truth_delta_y[b, k]
                    truth_response[b, position, ind, 2:4] = truth_cboxes[b, k, 2:4].log() / anchors[0, 0, ind, :]
                    truth_response[b, position, ind, 4] = max_iou
                    truth_response[b, position, ind, label_ind] = 1.0
                    truth_response_size[b, position, ind, :] = truth_cboxes[b, k, 2:4]
            
            mask_tbox_response = mask_response[:,:,:,0:5].contiguous().view(-1,5)
            mask_class_response = mask_response[:,:,:,5:].contiguous().view(-1,class_num)
            truth_tbox_response = truth_response[:,:,:,0:5].contiguous().view(-1,5)
            truth_class_response = truth_response[:,:,:,5:].contiguous().view(-1, class_num)
            mask_noobj_response = mask_response[:,:,:,4].logical_not() & mask_noobj_response

        pred_obj_tboxes = pred_tboxes.view(-1,5)[mask_tbox_response].view(-1,5)
        truth_obj_tboxes = truth_tbox_response[mask_tbox_response].view(-1,5)
        pred_obj_class = pred_class.view(-1, class_num)[mask_class_response].view(-1, class_num)
        truth_obj_class = truth_class_response[mask_class_response].view(-1, class_num)
        pred_noobj_probs = pred_tboxes[:,:,:,4].view(-1)[mask_noobj_response.view(-1)].view(-1)
        truth_noobj_probs = (pred_noobj_probs * 0.0).detach()
        # use coord_fix_rate to fix problem of 'larger box with larger loss'
        truth_obj_sizes = truth_response_size.view(-1,2)[mask_tbox_response[:,2:4].view(-1,2)].view(-1,2)
        coord_fix_rate = (2 - truth_obj_sizes[:,0] * truth_obj_sizes[:,1]).repeat(4, 1).permute(1,0).sqrt()

        loss_prior = self.prior_rate * (self.mse_loss(pred_cboxes[:,:,:,2:4].view(-1,2), anchors.view(-1,2), reduction='sum') if iteration < self.prior_iter else 0)
        loss_noobj = self.noobj_rate * self.mse_loss(pred_noobj_probs, truth_noobj_probs, reduction='sum')
        loss_coord = self.coord_rate * self.mse_loss(coord_fix_rate * pred_obj_tboxes[:,0:4], coord_fix_rate * truth_obj_tboxes[:,0:4], reduction='sum')
        loss_object = self.object_rate * self.mse_loss(pred_obj_tboxes[:,4], truth_obj_tboxes[:,4], reduction='sum')
        loss_class = self.class_rate * self.mse_loss(pred_obj_class, truth_obj_class, reduction='sum')
        loss_total = loss_noobj + loss_prior + loss_coord + loss_object + loss_class

        return loss_total / batch_size


if __name__ == "__main__":
    batch_size = 72
    preds = torch.rand((batch_size,14,14,125))
    preds.requires_grad = True
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


