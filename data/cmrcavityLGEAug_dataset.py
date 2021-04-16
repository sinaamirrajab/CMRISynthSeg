"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 0}

# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 0,'LV_Blood': 1, 'Scar': 2,'NO_reflow': 3}
# sina feb 2021 for the heart with separated  labels
# TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 8, 'LV_Myocardium': 9,'RV_Bloodpool': 10,'abdomen': 4,'Body_fat': 2,'vessel': 6, 'extra_heart': 1, 'Lung': 5,'Skeletal': 3}
# TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3,'abdomen': 4,'Body_fat': 5,'vessel': 6, 'extra_heart': 7, 'Lung': 8 ,'Skeletal': 9}


class CmrcavityLGEAugDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
       
        parser.set_defaults(label_nc=5)
        parser.set_defaults(output_nc=1)
        # parser.set_defaults(crop_size=128)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(add_dist=False)
        

        parser.add_argument('--label_dir', type=str, required=False, default = '/data/sina/dataset/LGE_emidec/AI_image_data/3D_patients/with_normalization',
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=False, default ='/data/sina/dataset/LGE_emidec/AI_image_data/3D_patients/with_normalization' ,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
                        
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """
        datalist = sorted(os.listdir(os.path.join(opt.image_dir)))

        imgroot =[] 
        mskroot =[]
        imglist =[]
        msklist =[]
        filename_pairs = []

        assert os.path.exists(opt.label_dir), 'list of masks  doesnt exist'
        assert os.path.exists(opt.image_dir), 'list of images doesnt exist'

        if opt.altered_anatomy:
            img_list = sorted(os.listdir(os.path.join(opt.image_dir)))
            msk_list = sorted(os.listdir(os.path.join(opt.label_dir)))
            filename_pairs = [(os.path.join(opt.image_dir,x),os.path.join(opt.label_dir, y)) for x,y in zip(img_list,msk_list)]
        else:
            for i in range(len(datalist)):
                imgroot+= [os.path.join(opt.image_dir,datalist[i],'Images')]
                mskroot += [os.path.join(opt.image_dir,datalist[i],'Contours')]
                imglist += sorted(os.listdir(os.path.join(opt.image_dir,datalist[i],'Images')))
                msklist += sorted(os.listdir(os.path.join(opt.image_dir,datalist[i],'Contours')))
                filename_pairs += [(os.path.join(imgroot[i],imglist[i]),os.path.join(mskroot[i], msklist[i]))]

        self.img_list = imglist
        self.msk_list = msklist
        self.filename_pairs = filename_pairs


    def initialize(self, opt):
        self.opt = opt
        self.get_paths(opt)
        if opt.phase == 'train':
            train_transforms = Compose([
                # cmr_tran.Resample(1.47,1.47), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                cmr_tran.RandomHorizontalFlip2D(p=0.5),
                cmr_tran.RandomVerticalFlip2D(p=0.5),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                # cmr_tran.Resample(1.47,1.47),
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        self.cmr_dataset = cmr.MRI2DSegmentationDataset(self.filename_pairs, transform = train_transforms, slice_axis=2)
        
        size = len(self.cmr_dataset)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        data_input = self.cmr_dataset[index]

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = data_input["gt"] # the label map equals the instance map for this dataset
        if not self.opt.add_dist:
            dist_tensor = 0
        input_dict = {'label': data_input['gt'],
                      'instance': instance_tensor,
                      'image': data_input['input'],
                      'path': data_input['filename'],
                      'gtname': data_input['gtname'],
                      'index': data_input['index'],
                      'segpair_slice': data_input['segpair_slice'],
                      'dist': dist_tensor
                      }

        return input_dict
    
    def __len__(self):
        return self.cmr_dataset.__len__()