
########
# INPORTS
########

from tabnanny import verbose
from dataloader_semantic import *
#from retele import *
from train2_semantic import *

import torch
torch.manual_seed(0)
import torchvision # contine data loader-uri pentru seturi de date comune
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np


# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru incarcarea retelelor:
from load_checkpoint import load_complete_model

# pentru punerea optimizatorului pe GPU:
from optimizer_to_device import optimizer_to

###### import-uri pentru retele
#from networks import *
from networks_folder_semantic.unet_semantic_brats_2D_v3_COPIE import UNET_semantic_brats_2D_v3_COPIE
from networks_folder_semantic.unet_semantic_brats_2D_v4_pls import UNET_semantic_brats_2D_v4

from networks_folder_semantic.segnet_semantic_brats_2D_v1 import Segnet_semantic_brats_v1

######### pentru transfer LR
from networks_folder_semantic.unet_semantic_brats_2D_v3_transferLR_1 import get_modified_network

# reteaua originala
from networks_folder_semantic.unet_semantic_brats_2D_v3_COPIE import UNET_semantic_brats_2D_v3_COPIE


def main():

    # pentru optimizatori
    import torch.optim as optim
    
    ########
    # CE TINE DE CUDA
    ########

    # eliberare cache
    torch.cuda.empty_cache()
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    ########
    # PATHS
    ########

    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\PNG_Validation_Dataset'
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Validation_Dataset'
    SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\models_test_semantic' # * de aici schimb path-ul
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\models_test_semantic\model26.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\FULLDTS_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61_0.878.pth'
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\models_test_semantic\model77.pth'

    #### Pentru dataset simplu full
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Test_Dataset' # * de aici schimb path-ul

    # @@@@@@ ASTA SE ALEGE @@@@@
    #### Pentru dataset GibbsArtRemoverd full
    TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'
    VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Val_Dataset'
    TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset' # * de aici schimb path-ul

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! E FOLDERUL DE TESTT
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset'

    ######################################### DATADET REDUS #########################################

    #### Pentru dataset simplu REDUCED 1/3 NEPREPROCESAT 8b
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Test_Dataset' # * de aici schimb path-ul

    #### ULTRAREDUCED
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Training_Dataset'

    #### Pentru dataset simplu REDUCED 1/3 NEPREPROCESAT 16b
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Test_Dataset' # * de aici schimb path-ul

    #### Pentru dataset simplu REDUCED 1/3 BIASSFIELDREMOVED 8b
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Test_Dataset' # * de aici schimb path-ul
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Val_Dataset'

    #### Gibbs ringing asrtifact removal
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Test_Dataset' # * de aici schimb path-ul
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Val_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Val_Dataset'



    #### ABSOLUTE FULL DATASET
    TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\ABSOLUTE_FULL_DATASET_tosendforrofficialvalidation'
    VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset'

    ########
    # CREATING DATA GENERATORS
    ########



    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 6,
              'shuffle': True,
              'num_workers': 1,
              'drop_last': True
              }
    #nr_epochs = 50 # * de aici se schimba nr de epoci
    nr_epochs = 100

    # s einstantiaza daatseturile
    #training_set = Dataset(TRAIN_DATA_PATH, normalize='0_mean_1std', zoom = False, rotate = False)
    #val_set = Dataset(VAL_DATA_PATH, normalize='0_mean_1std', zoom = False, rotate = False)

    training_set = Dataset(TRAIN_DATA_PATH, normalize=None, zoom = True, rotate = False, horiz_flip=True, vert_flip=True, gauss_disp=0.01)
    val_set = Dataset(VAL_DATA_PATH, normalize=None, zoom = False, rotate = False, horiz_flip=False, vert_flip=False, gauss_disp=0)

    # se instantiaza generatorii:
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    val_generator = torch.utils.data.DataLoader(val_set, **params)

    ########
    # SELECT NETWORK
    ########

    # varianta the best
    net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features = [32, 64, 128, 256]) # MAX: 0.929 pe lr=10^(-5)sau -4??? 
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=4, features = [64, 128, 256, 512])

    # se asigura precizia double a parametrilor
    # net = UNET_binar_brats_2D_v2(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # diferenta infima

    # Varianta cu + in loc de skip connectionuriu. Unet V4
    #net = UNET_semantic_brats_2D_v4(in_channels=4, out_channels=3, features = [32, 64, 128, 256])
    #net = UNET_semantic_brats_2D_v4(in_channels=4, out_channels=3, features = [64, 128, 256])

    # Varianta SegNet
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=3, features = [32, 64, 128, 256])
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=3, features = [64, 128, 256]) # best sofar
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=3, features = [64, 128, 256, 512]) # dezastrtuos
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=3, features = [64, 128])
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=3, features = [128, 256, 512])
    
    # pentru transfer lr
    '''
    # 1. Test
    net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=1, features = [32, 64, 128, 256])
    net = get_modified_network(net, verboise=False)
    '''

    #net = net.double()


    ########
    # SELECT OPTIMIZER
    ########
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001) # practic nu functioneaza
    optimizer = optim.Adam(net.parameters(), lr=0.0001) # ! e mai bine, cel putin primele 40 de epoci
    #optimizer = optim.Adam(net.parameters(), lr=0.00001) # cu 10^(-5) se antrena ok pe UNET_simplu
    #optimizer = optim.Adam(net.parameters(), lr=0.000001) # e mai rau

    ########
    # SELECT SCHEDULER
    ########
    # 1) ExponentialLR:
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # scheduler-ul modifica LR-ul, in timpul antrenarii

    # 2) StepLR: la fiecare step_size epoci, lr-ul se inmulteste (decay) cu gamma.
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose = True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, verbose = True)
    # IDEE DE BAZA: AR TREBUI SA SCAD LR-UL DOAR ATUNCI CAND LOSS-UL NU MAI SCADE. LA EPOCA IN CARE LOSS-UL SE PLAFONEAZA, SE SCADE LR-ul. NU ARE SENS LA 20 DE EPOCI - ATAT DE DES
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose = True)



    ########
    # CE TINE DE CUDA
    ########
    current_best_score = 0 # in cazul in care inca nu s-a intrat in functia load_complete_model(), acuratetea initiala e 0%
    # ma asigur ca se incepe de la epoca 0. Va fi suprascris daca se incarca modelul
    start_epoch = 77
    #net.to(device)

    ########
    ## Incarcare model
    ########
    net, optimizer, start_epoch, loss, current_best_score = load_complete_model(net, optimizer, LOAD_PATH)
    net.to(device)
    # optimizatorul este pus pe CPU dupa ce e incarcat de functia load_complete_model
    optimizer = optimizer_to(optimizer, device)
    # modificarea lr
    
    for g in optimizer.param_groups:
        g['lr'] = 0.00001 # se seteaza lr la 10^(-5)
    

    #net = train_network(net, device, training_generator, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_acc, mod = 'complete_save')
    net = train_network(net, device, training_generator, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_score, start_epoch, mod = 'complete_save') # MAX:0.93






if __name__ == '__main__':
    main()







