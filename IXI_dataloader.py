import pandas as pd

def get_IXI_dataloader(excel_path):
    df = pd.read_excel(excel_path)
    df = df[~df['AGE'].isnull()].reset_index(drop=True)
    df = df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)

    IDs = df['IXI_ID'].tolist()

    paths = []
    ages = []
    nii_path = os.path.join(os.getcwd(),'IXI_NII')
    for f in os.listdir(nii_path):
        ID = f[:-3]
        row = df[df['IXI_ID'].astype(int)==int(ID)]
        age = int(row['AGE'])
        paths.append(os.path.join(nii_path, f))
        ages.append(age)

    DF = pd.DataFrame({'file_name':paths,'Age':ages}).to_csv(os.path.join(os.getcwd(),'IXI.csv'), index=False)
