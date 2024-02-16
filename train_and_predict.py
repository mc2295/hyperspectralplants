
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import ImageDataLoaders, Transform
from models import Inception2D, Inception3D, EfficientNet2D, EfficientNet3D
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import tifffile
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

dic_label = {k : i for i,k in enumerate(['Healthy','Mild', 'YR', 'YR_Mild', 'Sept', 'YR_Sept'])}

class CustomImageDataset(Transform):
    def __init__(self , df, valid):
        if valid:
            self.df = df.loc[df['is_valid'] == True]
        else: 
            self.df = df.loc[df['is_valid'] == False]
        
    def __getitem__(self, idx):
        img = torch.tensor(tifffile.imread(self.df.name.iloc[idx]), device = 'cuda').permute(2,0,1).float()
        img = F.interpolate(img.unsqueeze(0), size = (299,299), mode = 'bilinear', align_corners=False).squeeze()
        return img, dic_label[self.df.label.iloc[idx]]
    def __len__(self):
        return len(self.df)


skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
batch_size = 6

df = pd.read_csv('df_all_images.csv')
df_result = pd.DataFrame(columns = ['y_true', 'y_preds'])

for k, (train_index, val_index) in enumerate(skf.split(df.index, df.label)):
    # df.assign(is_valid = False)

    criterion = torch.nn.CrossEntropyLoss()


    df['is_valid'] = False
    df.loc[df.index.isin(val_index), 'is_valid'] = True

    train_dataset =  CustomImageDataset(df, valid = False)
    valid_dataset = CustomImageDataset(df, valid = True)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)
    
    model = EfficientNet2D()
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9) # we tested for lr = 1e-2, lr= 1e-3, lr = 1e-4
    print(len(train_dataloader))
    print(len(valid_dataloader))
    
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            print(i)
            inputs,labels = data
            labels = labels.cuda()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            accuracy = accuracy_score(labels.cpu(), outputs.argmax(dim=1).cpu())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print('epoch : ', epoch, 'loss : ', loss, 'acc :', accuracy)
    accuracy_list = []
    y_true = []
    y_pred = []
    for i, data in enumerate(valid_dataloader): 

        inputs,labels = data
        y_true = y_true + labels.tolist()
        outputs = model(inputs)
        y_pred = y_pred + outputs.argmax(dim=1).cpu().tolist()
    print('final accuracy ', accuracy_score(y_true, y_pred))
    result = {'y_true': y_true, 'y_preds': y_pred}
    df_result.loc[len(df_result)] = result
    df_result.to_csv('model/efficient_result_2D_lr_2.csv')



    torch.save(model, 'model/inception_2D_lr_2_' + str(k))
