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
sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

Generate a title from 
http://patorjk.com/software/taag/#p=display&f=Doom&t=mricnn_predict

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
    def load_test_data():
   
    def post_process_data():
         
    def predict(self,options):



        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        create_test_data(options)

        imgs_test = load_test_data(options)
        imgs_test = imgs_test.astype('float32')


        imgs_test /= 255.  # scale masks to [0, 1]


        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)

        model = self.get_unet()
        weight_dir =options.inputdir+ '/weights'
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

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

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Mricnn_predict()
    chris_app.launch()
