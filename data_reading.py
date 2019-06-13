import pandas as pd

def read_dataframe(dirname):
	df = pd.read_csv(dirname)
	return df

pdf = read_dataframe('./datasets/amazon_google_exp_data/tableA.csv')
print(pdf.head())
pdf['total'] = pdf['title']+pdf['manufacturer']+str(pdf['price'])
print(pdf)