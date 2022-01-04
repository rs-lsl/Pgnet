from torch.utils.data import Dataset

class Mydata(Dataset):
    def __init__(self, lrhs, pan, label):
        super(Mydata, self).__init__()
        self.lrhs = lrhs
        self.pan = pan
        self.label = label

    def __getitem__(self, idx):
        assert idx < self.pan.shape[0]
        return self.lrhs[idx, :, :, :], self.pan[idx, :, :, :], self.label[idx, :, :, :]

    def __len__(self):
        return self.pan.shape[0]
