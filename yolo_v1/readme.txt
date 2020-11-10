# YOLO v1 Experiment (14x14x30, Resnet)
this is an experiment for YOLO v1

### DISCRIPTIONS

#### About code:
all code rewritten from https://github.com/xiongzihua/pytorch-YOLO-v1/, but very very differrent from the origin code. actually I am a beginner in using pytorch, write learning machine etc., so the origin code really helped me a lot. first it shows how the basic network be formed, second when I write a model myself, I can compare it with the origin code and more easily find the problem (almost everytime), third the origin code reminds me when I have no idea write something. also the pre-process for training image is fasinating.
but I also want to say that the origin code is not fully organized, and variable names are sometimes weird like 'coo_mask', which means 'coordinates mask' litarally based on the paper, but actually means 'has object mask'. comments are in chaos, code is in different styles. and if you search for some sentence, you would find it appears in many blogs. the key disadvantage is that the origin code can also misleading others. dispite of those 'not correct variable names', 'not correct comments' and 'not consistant approach (like iou calculation)', some approaches is also misleading. take yolo loss as an example: origin code make calculation of 'coo_response_mask' (actually means has object and responsible mask) in cpu and turn it into gpu later. one problem is that the code can only run in gpu 0, but the fatal misadvize is that make me think these code must be running in cpu and then turn into gpu, as I have failed again and again try to even only generate the mask in gpu, rather to say add something into it. but now I found the true matter, the autograd, we should try to prevent the autograd for the mask, but with no connection with using cpu or gpu. 

#### About target:
since h,w,d = cv2.read but not w,h,d, so target should be target[y,x,:]

#### About torch.autograd:
do not use b = a, that would make torch.autograd confused. that means if yu use b=a, then you modified b later, that would modify a, that means b is only a citation of a, they share the same memory 

if using truths_conf = preds_conf * 0.0; truths_conf[i] = ..., that would make unstable error. I think maybe, just maybe, if x * 0 would be calculated when backward also, that lead to this unstable. (x * 0).detach() maybe a way to solve it (this means the variable can not be autograded, then it would be seen as a constant). but I have not found any evidence yet, since d = a * 0 + a * a can be correctly autograded.

#### others
tensor runs slower than numpy
all torch using gpu would use gpu 0 as the basis data storage, though may not calc in it

#### loss 
loss using paper's method would train slower but more stable. results of two kind of loss method is stored in folder 'results', all of them have map > 60%

### EVIRONMENT
1. ubuntu 16.04.7 LTS  GNU/LINUX 4.15.0 x86-64
2. Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
3. Tesla K80, Cuda: 10.0 
3. (*.__version__) python 3.7.2, torch 1.4.0, cv2 4.4.0, numpy 1.19.2, torchvision 0.5.0, other packages...


### PREPARATIONS
1. Download voc2012train dataset
2. Download voc2007test dataset
3. put all images in one folder called 'allimgs' 
4. (not necessary) you can also generate your txt using functions in re_utils.py rather than using those prepared with 'voc20' as its prefix)


### TRAIN
(you can run python train.py -h first)
nohup python -u train.py -l tmp_loss_save.pkl -m tmp_model_save.pth -d 0123 2>&1 &  
watch -n 0.2 -d nvidia-smi


### EVALUATE
(you can run python eval.py -h first, 16 items/s)
python eval.py -m *.pth -d 4 

