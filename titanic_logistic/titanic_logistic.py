import pandas as pd 
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
total_data = pd.concat([train_data, test_data], ignore_index=True)

labelencoder = LabelEncoder()
total_data['Sex'] = labelencoder.fit_transform(total_data['Sex'])

total_data['Age'] = total_data['Age'].fillna(-1)
total_data.loc[total_data['Age'] <= 16, 'Age'] = 1
total_data.loc[total_data['Age'] > 16, 'Age'] = 2
total_data['Age'] = total_data['Age'].astype(int)

total_data.loc[(total_data['SibSp'] == 1) | (total_data['SibSp'] == 2), 'SibSp'] = 1
total_data.loc[total_data['SibSp'] > 2, 'SibSp'] = 2
total_data['SibSp'] = total_data['SibSp'].astype(int)

total_data['Parch_cut'] = pd.cut(total_data['Parch'], bins=[-1, 0, 3, 9], labels=[1, 2, 4])
total_data['Parch'] = total_data['Parch_cut'].astype(int)
total_data = total_data.drop(columns=['Parch_cut'])

total_data['Fare_cut'] = pd.cut(total_data['Fare'], bins=[-1, 15, 50, 1000], labels=[1, 2, 3])
total_data['Fare'] = total_data['Fare_cut'].fillna(1).astype(int)
total_data = total_data.drop(columns=['Fare_cut'])

total_data['Embarked'] = total_data['Embarked'].fillna('S')
total_data['Embarked'] = labelencoder.fit_transform(total_data['Embarked'])

train_x = total_data[total_data['Survived'].notnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
train_y = total_data[total_data['Survived'].notnull()]['Survived'].values
test_x = total_data[total_data['Survived'].isnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
test_x = torch.tensor(test_x, dtype=torch.float32)

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

class TitanicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = len(x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

train_set = TitanicDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_set, batch_size=100, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(train_x.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0
epoch = 2000
n_batch = len(train_loader)

for i in range(epoch):
    model.train()
    for j, (samples, labels) in enumerate(train_loader):
        pred = model(samples)
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (j + 1) % n_batch == 0:
            print(f"Epoch [{i + 1}/{epoch}], Batch [{j + 1}/{n_batch}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        train_preds = model(train_x).round()
        valid_preds = model(valid_x).round()
        
        correct_train = (train_preds == train_y).sum().item() / len(train_y)
        correct_valid = (valid_preds == valid_y).sum().item() / len(valid_y)
        
        print(f"Train Acc: {correct_train:.2%}, Valid Acc: {correct_valid:.2%}")
        
        if correct_valid > best_acc:
            best_acc = correct_valid
            torch.save(model.state_dict(), 'model.pth')
            print("Model saved with best validation accuracy:", best_acc)

model.load_state_dict(torch.load('model.pth'))

valid_y_np = valid_y.numpy().astype(int)
valid_preds_np = valid_preds.numpy().astype(int)

print(f"Train Acc: {correct_train:.2%}, Valid Acc: {correct_valid:.2%}")
        
print("\nConfusion Matrix:")
print(confusion_matrix(valid_y_np, valid_preds_np))

with torch.no_grad():
    predictions = model(test_x).round().view(-1).numpy().astype(int)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    submission.to_csv('IT_submission.csv', index=False)
