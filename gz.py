import os
import gzip
import shutil

######################
# 2) compreseaza .gz-urile
######################




DIR_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_submission_NII\output_ABSOLUTE_FULL_DTS__unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph68"
OUTPUT_DIR_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_submission_NII\output_ABSOLUTE_FULL_DTS__unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph68_niiuri"

for file_name in os.listdir(DIR_PATH):
    FILE_PATH = DIR_PATH + '\\' + file_name
    NEW_FILE_PATH = OUTPUT_DIR_PATH + '\\' + file_name + '.gz'
    with open(FILE_PATH, 'rb') as f_in:
        with gzip.open(NEW_FILE_PATH, 'wb') as f_out:
            f_out.write(f_in.read())


######################
# 2) se sterg .nii arhivate
######################
'''
for file_name in os.listdir(DIR_PATH):
    FILE_PATH = DIR_PATH + '\\' + file_name
    if FILE_PATH.split('.')[-1] == 'gz':
        os.remove(FILE_PATH)
'''