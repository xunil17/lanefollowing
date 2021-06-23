from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

real_driving = '/home/sean/tensorflow_datasets/driving_dataset/driving_dataset/data.txt'
file_path = '/home/sean/lanefollowing/ros2_ws/src/lane_following/train/data/training_data.csv'

df = pd.read_csv(file_path, delimiter=',', header=None, index_col=0)

val_lst = df.loc[:,1].tolist()

plt.figure()
plt.hist(val_lst)

plt.show()