import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from eval_metric.metrics_classification import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14): 
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return

    
def revert_sync_batchnorm(module):            # convert each SyncBatchNorm to BatchNormXd
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

class Dino_Xcit(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        
        
        self.backbone = backbone
        
        self.classifier = classifier 
         
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.backbone(x)
        x = self.classifier(x) # 
        
        return x

if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16', pretrained=True) # load the encoder
    backbone=revert_sync_batchnorm(backbone)  # convert each SyncBatchNorm to BatchNormXd

    state_dict = torch.load("pretrained_weights/dino_xcit_small_12_p16_linearweights.pth")['state_dict'] # load the weights of the pretrained classifier and rename the keys
    state_dict = {k.replace("module.linear.", ""): v for k, v in state_dict.items()}
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(384, 1000)) # building the classifier of the model
    classifier[1].load_state_dict(state_dict) # load the weights dino_xcit_small_12_p16_linearweights to the builded classifier

    model = Dino_Xcit(backbone, classifier) # build the model
                      
    model.eval()
    

    if args.use_cuda:
        model = model.cuda()
        

    target_layers = [model.backbone.cls_attn_blocks[-1].norm1] # or model.backbone.blocks[-1].norm2 

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))   # 
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 186
    #tiger shark = 3
    # bald eagle = 22
    # Norwich terrier = 186

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    masked_tensor = preprocess_image(cam_image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    #Parameters needed for the evaluation metrics
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
     
    if args.use_cuda:
        device = 'cuda'
        input_tensor=input_tensor.cuda()
        blur = lambda x: nn.functional.conv2d(x.to("cpu"), kern, padding=klen//2).to("cuda") # Function that blurs input image 
    else:
        device = 'cpu'
        blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2) # Function that blurs input image
    
    # Γενικότερα για target category = None έχoυν πρόβλημα οι παρακάτω συναρτήσεις. Να το δώ
    outputs, outputs_mask = return_probs(model, input_tensor, masked_tensor, device)
    print("For unmasked input, the model predicted", outputs[1][0][0].item(), "class with probability", outputs[0][0][0].item())
    print("For masked input, the model predicted", outputs_mask[1][0][0].item(), "class with probability", outputs_mask[0][0][0].item())
    prob1, prob2 = return_probs2(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
    print("The probabilities for target class", target_category,"and for unmasked input is", prob1, ",while for masked input", prob2)
    avg_drop, avg_inc = averageDropIncrease(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
    print("The AVERAGE DROP is", avg_drop)
    print("Is score for masked input higher?", avg_inc)
    
    #Starting deletion and insertion game 
    
    insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

    del_scores = deletion.single_run(input_tensor, cam_image, verbose=1, save_to = "plots/del224.png")
    
    in_scores = insertion.single_run(input_tensor, cam_image, verbose=1, save_to = "plots/ins224.png")
    
    del_auc = auc(del_scores) # Calculation of AUC scores
    in_auc = auc(in_scores)
    print("The AUC score for deletion is", '%.3f' % del_auc,"and for insertion", '%.3f' % in_auc)
    
    #Implementation of Energy-based Pointing Game proposed in Score-CAM
    #bbox = [32, 71, 177, 197] # bbox for bald_eagle 224*224
    #bbox = [30, 42, 194, 172] # bbox for tiger_shark 224*224
    bbox = [5, 2, 217, 222] # bbox for Norwich_terrier 224*224

    proportion = energy_point_game2(bbox, torch.tensor(grayscale_cam))
    print("The PROPORTION after energy point game is", '%.3f' % proportion.item())
    
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)