import argparse
import cv2
import numpy as np
import torch
from torchvision import models
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
    deprocess_image, \
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
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
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

    model = models.resnet50()  # Moco-v3 architecture based on Resnet in evaluation mode is equal with Resnet architecture
    state_dict = torch.load("pretrained_weights/linear-1000ep.pth.tar")['state_dict'] # load the weights of the pretrained model and rename the keys
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict) 

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4[-1].conv3] # or model.layer4[-1]. For ScoreCam for target layer model.layer4[-1] we don't observe meaningful activations maps


    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224)) #
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 186
    #tiger shark = 3
    # bald eagle = 22
    # Norwich terrier = 186

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

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
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)