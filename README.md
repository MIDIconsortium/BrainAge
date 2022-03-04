# Accurate brain-age models for routine clinical MRI examinations

This repository contains scripts to enable readers to run the trained models presented in Wood et al., [Accurate brain-age models for routine clinical MRI examinations](https://www.sciencedirect.com/science/article/pii/S1053811922000015?via%3Dihub#fig0003), 2022, NeuroImage. The models were trained on > 23,000 axial T2-weighted head MRI scans from two large UK hospitals, and demonstrate accurate, generalisable brain-age prediction:

<img src="https://user-images.githubusercontent.com/67752614/152115266-02e2c8ff-2994-4115-a0bf-75d6cf576f7f.png" width=60% height=60%>



# Requirements

The code requires the data to be in Nifti file format (.nii or .nii.gz extension) and makes use of the [Project Monai library](https://monai.io/). This repository is compatible with python 3.6. See requirements.txt for all prerequisites; you can also install them using the following command:

`pip install -r requirements.txt`

# Usage

Running our models is straightforward. All that is needed is a .csv file with the following two columns:

- 'ID' (which is a unique identifier for each participant/scanning session e.g., 'pat119' etc.).

- 'file_name' (which gives the absolute path to the Nifti file for each participant).

Optionally, users can also provide a third column named 'Age', which gives the chronological age of each participant in years, in order to generate performance metrics (e.g., mean absolute error (MAE)) and scatter plots). 

Brain-age prediction can then be performed using the following command:

`python run.py --project_name NAME --csv_file /PATH/TO/CSV/FILE`.

This will save a .csv file within the local cloned repository (./NAME_output.csv) with the brain-predicted ages for each subject. If a scatter-plot is required, run.py should be called with the additional argument --return_metrics.

By default, our model will run on a cpu, and taskes ~ seconds to preprocess and return a brain-age prediction for each scan. If a GPU is available, run.py should be called with the additional argument --gpu

By default, run.py assumes axial T2-weighted scans are provided. If instead axial diffusion-weighted scans are provided, then run.py should be called with the additional argument --sequence dwi

Please note that our model only provides meaningful brian-age predictions for scans that are oriented in the 'LPS' coordinate system (i.e., right to **L**eft, anterior to **P**osterior, inferior to **S**uperior). For this reason, run.py automatically reorients scans to this coordinate system.

### Running models with Information eXtraction from Images (IXI) dataset
To reproduce the results on the open-access IXI dataset, first download (and unzip) the axial T2-weighted scans and associated .csv file [here](https://brain-development.org/ixi-dataset/). Brain-age prediction can then be performed using the following command:

`python run.py --ixi --project_name IXI --csv_file /home/dw19/Downloads/IXI_file.xls --ixi_nii_dir /PATH/TO/IXI/NII/FILES --return_metrics`

This will save a .csv file with brain-predicted ages for each IXI participant, along with the following scatter plot:


<img src="https://user-images.githubusercontent.com/67752614/152117840-580e1afa-477c-46cc-9778-f63b0c4fd961.png" width=35% height=35%>

Note that /PATH/TO/IXI/NII/FILES must be the path to the extracted files after unzipping 'IXI-T2.tar', and should contain files such as 'IXI002-Guys-0828-T2.nii.gz','IXI012-HH-1211-T2.nii.gz' etc.


# Coming soon

We will be releasing our 'skull-stripped' model which takes as input axial T2-weighted scans which have had non-brain-tissue removed. We will also be releaseing our volumetric T1-weighted models (raw and skull-stripped).

# Citation
If you found this repository useful, please consider citing our work:

```@article{wood2022accurate,
  title={Accurate brain-age models for routine clinical MRI examinations},
  author={Wood, David A and Kafiabadi, Sina and Al Busaidi, Ayisha and Guilhem, Emily and Montvila, Antanas and Lynch, Jeremy and Townend, Matthew and Agarwal, Siddharth and Mazumder, Asif and Barker, Gareth J and others},
  journal={NeuroImage},
  pages={118871},
  year={2022},
  publisher={Elsevier}
}```
