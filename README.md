# Accurate brain-age models for routine clinical MRI examinations

This repository contains scripts to enable readers to run the trained models presented in [Accurate brain-age models for routine clinical MRI examinations](https://www.sciencedirect.com/science/article/pii/S1053811922000015?via%3Dihub#fig0003), (Wood et al., 2022, NeuroImage) and [Optimising brain age estimation through transfer learning: a suite of pre-trained foundation models for improved performance and generalisability in a clinical setting](https://onlinelibrary.wiley.com/doi/10.1002/hbm.26625), (Wood et al., 2024, Human Brain Mapping). The models were trained on > 23,000 axial T2-weighted head MRI scans from two large UK hospitals, and demonstrate accurate, generalisable brain-age prediction:

<img src="https://user-images.githubusercontent.com/67752614/152115266-02e2c8ff-2994-4115-a0bf-75d6cf576f7f.png" width=60% height=60%>



# Requirements

The code requires the data to be in Nifti file format (.nii or .nii.gz extension) and makes use of the [Project Monai library](https://monai.io/). This repository is compatible with python 3.6. See requirements.txt for all prerequisites; you can also install them using the following command:

`pip install -r requirements.txt`

# Usage

Running our models is straightforward. All that is needed is a .csv file with the following two columns:

- 'ID' (which is a unique identifier for each participant/scanning session e.g., 'pat119' etc.).

- 'file_name' (which gives the absolute path to the Nifti file for each participant).

Brain-age prediction can then be performed using the following basic command:

`python run_inference.py --project_name NAME --csv_file /PATH/TO/CSV/FILE`

This will save a .csv file within the local cloned repository (./NAME_output.csv) with the brain-predicted ages for each subject. Optionally, users can also provide a third column named 'Age', which gives the chronological age of each participant in years, in order to generate performance metrics (e.g., mean absolute error [MAE] and scatter plots). To do this, simply add the argument --return_metrics to run_inference.py.

By default, our model will run on a cpu. If a GPU is available, run.py should be called with the additional argument --gpu (in this case, run time is <2 seconds per scan).

To run our skull-stripped T2 model, simply add the argument --skull_strip to run_inference.py. Likewise, to use our volumetric T1-weighted ensemble model (which also relies on skull-stripping), then run_inference.py should be called with the following additional arguments:

--sequence t1
--ensemble 

Note skull-stripping is performed using [HD-BET](https://github.com/MIC-DKFZ/HD-BET) - a deep-learning based brain extraction tool which takes ~10 seconds per scan with a gpu and ~ 1 minute without.

To run our axial diffusion-weighted, axial susceptibility-weighted, coronal flair or sagittal T1-weighted models, then run_inference.py should be called with the following additional arguments:

-- dwi

-- swi

-- cor_flair

-- sag_T1

Please note that our model only provides meaningful brian-age predictions for scans that are oriented in the 'LPS' coordinate system (i.e., right to **L**eft, anterior to **P**osterior, inferior to **S**uperior). For this reason, run_inference.py automatically reorients scans to this coordinate system.



# Citation
If you found this repository useful, please consider citing our work:

Wood, D. A., Kafiabadi, S., Al Busaidi, A., Guilhem, E., Montvila, A., Lynch, J., ... & Booth, T. C. (2022). Accurate brain-age models for routine clinical MRI examinations. NeuroImage, 118871. https://doi.org/10.1016/j.neuroimage.2022.118871

# Coming soon
We have recentely had a follow-up paper [Optimising brain age estimation through transfer learning: A suite of pre-trained foundation models for improved performance and generalisability in a clinical setting](https://onlinelibrary.wiley.com/doi/10.1002/hbm.26625) published in Human Brain Mapping. The code to run and fine-tune these foundation models will be added shortly!
