import argparse
import cv2
import numpy as np
import torch
import timm

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

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
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

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

    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 186

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
    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Γενικότερα για target category = None έχoυν πρόβλημα οι παρακάτω συναρτήσεις. Να το δώ
    outputs, outputs_mask = return_probs(model, input_tensor, masked_tensor, device)
    print("For unmasked input, the model predicted", outputs[1][0][0].item(), "class with probability", outputs[0][0][0].item())
    print("For masked input, the model predicted", outputs_mask[1][0][0].item(), "class with probability", outputs_mask[0][0][0].item())
    prob1, prob2 = return_probs2(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
    print("The probabilities for target class", target_category,"and for unmasked input is", prob1, ",while for masked input", prob2)
    avg_drop, avg_inc = averageDropIncrease(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
    print("The AVERAGE DROP is", avg_drop)
    print("Is score for masked input higher?", avg_inc)
    
    #Starting deletion and insertion game ONLY FOR CPU
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2) # Function that blurs input image  
    # the two previous lines will be used for insertion mode
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
