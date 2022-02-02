import pandas as pd


if __name__ == "__main__":
    path = os.path.join(os.path.getcwd(),'IXI.xls'))
    print(os.path.exists(path))
    df = pd.read_excel(path)
    df = df[~df['AGE'].isnull()].reset_index(drop=True)
    df = df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)

    print(df.head())
