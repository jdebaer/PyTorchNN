import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        return self.data[idx]

class NormDataset(Dataset):
    def __init__(self, subset, scaler_path=None, transform=None):
        self.data = subset[:]
        self.transform = transform

        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            self.data = self.scaler.transform(self.data) 

        else:
            # Note: MinMaxScaler(feature_range = (0, 1)) will transform each value in the column proportionally within the range [0,1]. 
            # of [-1,1] if there are neg values.
            # Use this as the first scaler choice to transform a feature, as it will preserve the shape of the dataset (no distortion).
            # MinMaxScaler does normalization.
            # StandardScaler() will transform each value in the column to range about the mean 0 and standard deviation 1, ie, each value 
            # will be normalised by subtracting the mean and dividing by standard deviation. Use StandardScaler if you know the data 
            # distribution is normal.
            # StandardScaler does not guarantee balanced feature scales, due to the influence of the outliers while computing the 
            # empirical mean and standard deviation.
            # By using RobustScaler(), we can remove the outliers and then use either StandardScaler or MinMaxScaler for preprocessing 
            # the dataset. 
            # If there are outliers, use RobustScaler(). Alternatively you could remove the outliers and use either of the above 2 scalers 
            # (choice depends on whether data is normally distributed)
            # scaler = sklearn.preprocessing.RobustScaler( 
            # with_centering=True, 
            # with_scaling=True, 
            # quantile_range=(25.0, 75.0), 
            # copy=True, 
            # ) 
            # robust_df = scaler.fit_transform(x)
            # if x was a DataFrame and not a Numpy array, you have to convert the output array back to a DataFrame:
            # robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2']) # whatever the cols are in the original dataframe.
            # Additional Note: If scaler is used before data split (train/valid/test), data leakage will happen. Do use scaler after splitting. 

            # normalize data and save scaler for inference with valid/test data
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data) #fits and transforms
            joblib.dump(self.scaler, "./scaler.pkl") 

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):
        input = self.data[idx, 0:6]
        label = self.data[idx, 6]
        sample = { 'input': input, 'label': label}
        return sample

        # sample["input"] == ndarray
        # sample["label"] == float 32
        
    def use_scaler_to_scale(self):
        pass


class CreateDataLoaders():
    def __init__(self, batch_size):
        # Can also works with Pandas dataframe.
        # Note: balance the data as much as possible if the distribution is not equal between classes.
        # This must happen before you split data
        # self.data_as_np = np.genfromtxt('./data.csv', delimiter=',', dtype=np.float32)
        self.df = pd.read_csv('./data.csv', index_col=False, dtype=np.float32)

        # Balance classes.
        # This is important for neural networks but also the hardest one to realize.

        ## Handle missing values - easiest way is to drop all rows with None, NaN or NaT.
        # self.df = self.df.dropna()


        # Remove outlier samples: MUST be done before normalization. 
        # Removing outliers should also be done for standardization, otherwise scales will be off.
        # RobustScaler can do this, see higher.

        ## One-hot encode pure categorical data.
        # cat_cols = self.df.select_dtypes(include=["object"]).columns.to_list()
        # self.df = pd.get_dummies(df, columns = cat_cols, dtype=int)
        ## For ordinal data like S/M/L we can do better with ordinal encoding to e.g., 1/2/3.

        ## Remove features (PCA). PCA requires standardization and not normalization since it assumes normal distribution of data.
        # from sklearn.decomposition import PCA
        # self.df_pca =(self.df - self.df.mean()) / self.df.std() # This is just so we can run PCA.
        # pca = PCA(n_components=5)
        # pca.fit(self.df_pca)

        # Then AFTER data split (train/valid/test):
        # Normalize your data == min-max scaling. For neural networks we usually do normalization and not standardization see below.
        #     Note: for ML algs that assume normal distr. in data such as Lin Regr. (but not NN) you should use standardization.
        # After all this you have a solid Pandas/Numpy array with training data.

        self.base_dataset = BaseDataset(self.df.to_numpy())

        #train_size = int(0.8 * len(self.base_dataset))
        #valid_size = len(self.base_dataset) - train_size
        train_size = int(0.7 * len(self.base_dataset))
        valid_size = int(0.2 * len(self.base_dataset))
        test_size = int(0.1 * len(self.base_dataset))

        split_size_total = train_size + valid_size + test_size
        if split_size_total == len(self.base_dataset):
            pass
        elif split_size_total == (len(self.base_dataset) -1):
            test_size = test_size + 1
        else:
            print("Fix alg. to calc dataset sizes.")
            exit(-1)

        train_dataset, valid_dataset, test_dataset = random_split(self.base_dataset, [train_size, valid_size, test_size])

        train_dataset_norm = NormDataset(train_dataset)
        valid_dataset_norm = NormDataset(valid_dataset, scaler_path="./scaler.pkl")
        test_dataset_norm = NormDataset(test_dataset, scaler_path="./scaler.pkl")

        self.train_loader = DataLoader(dataset=train_dataset_norm, batch_size=batch_size)
        self.valid_loader = DataLoader(dataset=valid_dataset_norm, batch_size=batch_size)
        self.test_loader = DataLoader(dataset=test_dataset_norm, batch_size=batch_size)

def main():
    batch_size = 16
    data_loaders = CreateDataLoaders(batch_size)

if __name__ == '__main__':
    main()
