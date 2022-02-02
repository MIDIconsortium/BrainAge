# Accurate brain-age models for routine clinical MRI examinations

This repository contains scripts to enable readers to run the trained models presented in Wood et al., [Accurate brain-age models for routine clinical MRI examinations](https://www.sciencedirect.com/science/article/pii/S1053811922000015?via%3Dihub#fig0003), 2022, NeuroImage. The models were trained on > 23,000 axial T2-weighted head MRI scans from two large UK hospitals, and demonstrate accurate, generalisable brain-age prediction:

![image](https://user-images.githubusercontent.com/67752614/152115266-02e2c8ff-2994-4115-a0bf-75d6cf576f7f.png)

The code requires the data to be in Nifti file format and makes heavy use of the [Project Monai library](https://monai.io/). This repository is compatible with python 3.6. See requirements.txt for all prerequisites; you can also install them using the following command:

`pip install -r requirements.txt`

# Usage

To reproduce the results on the open-access IXI dataset, first download (and unzip) the axial T2-weighted scans and associated .csv file [here](https://brain-development.org/ixi-dataset/). To be compatible for use with our models, these scans must first be proprocessed (resampled to 1mm^3 isotropic, cropped/padded etc.). This can be done using the following command:

`python IXI_preprocess.py IXI_directory path_to_IXI_csv`

Here, 'IXI_nii_directory' is the absolute path to the directory where the IXI scans in Nifti format have been unzipped to (this should contain files such as 'IXI497-Guys-1002-T2.nii' etc), whereas 'path_to_IXI_csv' is the location of the saved IXI csv file e.g., /home/user/Downloads/IXI.xls' etc.

IXI_preprocess.py will create a local directory ('IXI_nii' in the 'BrainAge' folder which is generated when cloning this repo) with pre-processed scans. Brain-age prediction can then be performed using the following command:

`python run_IXI.py`

This will save a .csv file with brain-predicted ages for each IXI participant, along with the following scatter plot:

![IXI_scatter](https://user-images.githubusercontent.com/67752614/152117840-580e1afa-477c-46cc-9778-f63b0c4fd961.png)

To run our brain-age models with other datasets, first pre-process the scans by running the following command:

`python preprocess.py path_to_csv`

This csv file should have the following columns:

- 'ID' which is a unique string identifier for each participant/scanning session (e.g., 'pat119' etc.)

- 'Age' which gives the chronological age of each participant in years

- 'file_name' which gives the absolute path to the Nifti file for each participant.

Again, this will create a local directory of pre-processed scans. Brain age prediction can then be performed using the following command:

`python run.py`

Again, this will save a .csv file with brain-predicted ages for each participant, along with a scatter plot.


# Coming soon

We will be releasing our 'skull-stripped' model which takses as input axial T2-weighted scans with non-brain-tissue removed.
