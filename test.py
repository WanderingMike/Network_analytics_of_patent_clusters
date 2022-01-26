import blobcity as bc
import pandas as pd

data = pd.read_csv("data/ready/ready_A43C.csv", index_col=0)
print(data)

model = bc.train(df=data, target="output")

model.features()
model.plot_feature_importance()


