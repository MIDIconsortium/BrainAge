import pandas as pd



df = pd.read_excel('https://github.com/MIDIconsortium/BrainAge/blob/main/IXI.xls')
df = df[~df['AGE'].isnull()].reset_index(drop=True)
df = df.drop_duplicates(subset='IXI_ID', keep=False).reset_index(drop=True)

print(df.head())
