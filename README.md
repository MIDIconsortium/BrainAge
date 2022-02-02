# Accurate brain-age models for routine clinical MRI examinations

This repository contains scripts to enable readers to run the trained models presented in Wood et al., "Accurate brain-age models for routine clinical MRI examinations", 2022, NeuroImage. The models were trained on > 23,000 axial T2-weighted head MRI scans from two large UK hospitals, and demonstrate accurate, generalisable brain-age prediction:

![image](https://user-images.githubusercontent.com/67752614/152115266-02e2c8ff-2994-4115-a0bf-75d6cf576f7f.png)

The code requires the data to be in Nifti file format and makes heavy use of the [Project Monai library](https://monai.io/). This repository is compatible with python 3.6. See requirements.txt for all prerequisites; you can also install them using the following command:

`pip install -r requirements.txt`

# Usage

To reproduce the results on the open access IXI dataset, first download (and unzip) the axial T2-weighted scans and associated .csv file [here](https://brain-development.org/ixi-dataset/). To be compatible for use with our models, these scans must be proprocessed (resampled to 1mm^3 isotropic, cropped/padded etc.). This can be done using the following command:

`python IXI_preprocess.py IXI_dir path_to_IXI_csv`

This will create a local directory (IXI_nii) containing pre-processed scans.

Brain-age prediction can then be performed using the following command:

`python run_IXI.py`

Once complete, a .csv file with predicted ages for each IXI participant will be saved locally, along with the following scatter plot:



![IXI_scatter](https://user-images.githubusercontent.com/67752614/152116497-c017631c-4a8b-4826-8825-2d79699cd954.png)



