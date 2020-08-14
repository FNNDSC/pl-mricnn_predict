#!/usr/bin/env python                                            
#
# mricnn_predict ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#


import os
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__))
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model,load_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D

# import the Chris app superclass
from chrisapp.base import ChrisApp
project_name = '3D-Dense-Unet'
img_rows = 256
img_cols = 256
img_depth = 16
smooth = 1.

Gstr_title = """

                _                                         _ _      _   
               (_)                                       | (_)    | |  
 _ __ ___  _ __ _  ___ _ __  _ __      _ __  _ __ ___  __| |_  ___| |_ 
| '_ ` _ \| '__| |/ __| '_ \| '_ \    | '_ \| '__/ _ \/ _` | |/ __| __|
| | | | | | |  | | (__| | | | | | |   | |_) | | |  __/ (_| | | (__| |_ 
|_| |_| |_|_|  |_|\___|_| |_|_| |_|   | .__/|_|  \___|\__,_|_|\___|\__|
                                ______| |                              
                               |______|_|                

"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       mricnn_predict.py 

    SYNOPSIS

        python mricnn_predict.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python mricnn_predict.py   \\
                                in    out

    DESCRIPTION

        `mricnn_predict.py` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit. 

"""


class Mricnn_predict(ChrisApp):
    """
    An app to predict segmented brain MRI images from a given unsegmented brain MRI.
    """
    AUTHORS                 = 'Sandip Samal (sandip.samal@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A neural network prediction app'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to predict segmented brain MRI images from a given unsegmented brain MRI'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('--model',dest='model',type=str,optional=False,
                          help='Which model do you want to train?')
        self.add_argument('--testDir',dest='testDir',type=str,default="test",optional=False,
                          help='Specify the name of the directory that contains the test data')

    def get_unet(self):
        inputs = Input((img_depth, img_rows, img_cols, 1))
        conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conc11 = concatenate([inputs, conv11], axis=4)
        conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
        conc12 = concatenate([inputs, conv12], axis=4)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

        conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conc21 = concatenate([pool1, conv21], axis=4)
        conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
        conc22 = concatenate([pool1, conv22], axis=4)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

        conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conc31 = concatenate([pool2, conv31], axis=4)
        conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
        conc32 = concatenate([pool2, conv32], axis=4)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

        conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conc41 = concatenate([pool3, conv41], axis=4)
        conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
        conc42 = concatenate([pool3, conv42], axis=4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

        conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conc51 = concatenate([pool4, conv51], axis=4)
        conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conc51)
        conc52 = concatenate([pool4, conv52], axis=4)

        up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
        conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conc61 = concatenate([up6, conv61], axis=4)
        conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc61)
        conc62 = concatenate([up6, conv62], axis=4)

        up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
        conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conc71 = concatenate([up7, conv71], axis=4)
        conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc71)
        conc72 = concatenate([up7, conv72], axis=4)

        up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
        conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conc81 = concatenate([up8, conv81], axis=4)
        conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc81)
        conc82 = concatenate([up8, conv82], axis=4)

        up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
        conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conc91 = concatenate([up9, conv91], axis=4)
        conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc91)
        conc92 = concatenate([up9, conv92], axis=4)

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc92)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()
        #plot_model(model, to_file='model.png')

        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def create_test_data(self,options):
        test_data_path= options.inputdir+'/%s/' %options.testDir
        dirs = os.listdir(test_data_path)
        total = int(len(dirs))*18

        imgs = np.ndarray((total, img_depth, img_rows, img_cols), dtype=np.uint8)

        i = 0
        j = 0
        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        for dirr in sorted(os.listdir(test_data_path)):
            dirr = os.path.join(test_data_path, dirr)
            images = sorted(os.listdir(dirr))
            count = total
            for image_name in images:
                img = imread(os.path.join(dirr, image_name), as_gray=True)
                #img= imread(dirr+'/'+image_name)
                #info = np.iinfo(img.dtype) # Get the information of the incoming image type
                img = img.astype(np.uint8)
                #img = img/255.

                img = np.array([img])
                if i< 17:
                    imgs[i][j] = img

                    j += 1

                    # if j % (image_depth-2) == 0:
                    #     imgs[0][i+1][0] = img
            
                    if j % (image_depth-1) == 0:
                        imgs[i][0] = img

                    if j % image_depth == 0:
                        imgs[i][1] = img
               	        j = 2
               	        i += 1
               	        if (i % 100) == 0:
                            print('Done: {0}/{1} test 3d images'.format(i, count))

        print('Loading done.')

        imgs = self.preprocess(imgs)

        np.save(options.inputdir+'/imgs_test.npy', imgs)

        imgs = self.preprocess_squeeze(imgs)

        count_processed = 0
        pred_dir = 'test_preprocessed'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs.shape[0]):
            for y in range(0, imgs.shape[1]):
                imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]*imgs.shape[1]))

        print('Saving to .npy files done.')


    def load_test_data(self,options):
        imgs_test = np.load(options.inputdir+'/imgs_test.npy')
        return imgs_test


    def preprocess(self,imgs):
        imgs = np.expand_dims(imgs, axis=4)
        print(' ---------------- preprocessed -----------------')
        return imgs

    def preprocess_squeeze(self,imgs):
        imgs = np.squeeze(imgs, axis=4)
        print(' ---------------- preprocessed squeezed -----------------')
        return imgs
   
         
    def predict(self,options):

        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        self.create_test_data(options)

        imgs_test = self.load_test_data(options)
        imgs_test = imgs_test.astype('float32')


        imgs_test /= 255.  # scale masks to [0, 1]


        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)

        model = self.get_unet()
        weight_dir =options.inputdir
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        model.load_weights(os.path.join(weight_dir, options.model + '.h5'))

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)

        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

        npy_mask_dir = 'test_mask_npy'
        if not os.path.exists(npy_mask_dir):
            os.mkdir(npy_mask_dir)

        np.save(os.path.join(npy_mask_dir, project_name + '_mask.npy'), imgs_mask_test)

        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)

        imgs_mask_test = preprocess_squeeze(imgs_mask_test)
        # imgs_mask_test /= 1.7
        #imgs_mask_test = np.around(imgs_mask_test, decimals=0)
        #info = np.iinfo(np.uint16) # Get the information of the incoming image type
        #imgs_mask_test = imgs_mask_test.astype(np.uint16)
        #imgs_mask_test=imgs_mask_test* info.max # convert back to original class/labels
        imgs_mask_test = (imgs_mask_test*255.).astype(np.uint8)
        count_visualize = 1
        count_processed = 0
        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        pred_dir = os.path.join(options.outputdir, project_name)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for x in range(0, imgs_mask_test.shape[0]):
            for y in range(0, imgs_mask_test.shape[1]):
                if (count_visualize > 1) and (count_visualize < 16):
                    save_img=imgs_mask_test[x][y].astype(np.uint16)
                    imsave(os.path.join(pred_dir, 'pred_' +str( count_processed )+ '.png'), save_img)
                    count_processed += 1

                count_visualize += 1
                if count_visualize == 17:
                    count_visualize = 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0]*14))

        print('-'*30)
        print('Prediction finished')
        print('-'*30)

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())
        self.predict(options)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Mricnn_predict()
    chris_app.launch()
