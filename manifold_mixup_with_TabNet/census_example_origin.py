"""
Requirements
"""
import pandas as pd
import os
import wget
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(0)
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)



"""
Download census-income dataset
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
dataset_name = 'census-income'
out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')

out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("File already exists.")
else:
    print("Downloading file...")
    wget.download(url, out.as_posix())


"""
Load data and split
"""
train = pd.read_csv(out)
target = ' <=50K'
np.random.seed(2023)
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index



"""
Simple preprocessing
"""
nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

# check that pipeline accepts strings
train.loc[train[target]==0, target] = "wealthy"
train.loc[train[target]==1, target] = "not_wealthy"



"""
Define categorical features for categorical embeddings
"""
unused_feat = ['Set']
features = [ col for col in train.columns if col not in unused_feat+[target]]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]



"""
Grouped features
"""
grouped_features = [[0,1,2], [8,9,10]]



"""
Network parameters
"""
tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":2,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax', # "sparsemax"
                 # "grouped_features" : grouped_features,
                 "device_name" : 'cuda:0'
                }

clf = TabNetClassifier(**tabnet_params)



"""
Training
"""
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]


max_epochs = 50 if not os.getenv("CI", False) else 2

from pytorch_tabnet_custom.augmentations import ClassificationSMOTE
aug = ClassificationSMOTE(p=0.2)


# This illustrates the warm_start=False behaviour
save_history = []
# for _ in range(2):
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    max_epochs=max_epochs , patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    # augmentations=aug, #aug, None
)
save_history.append(clf.history["valid_auc"])

# assert(np.all(np.array(save_history[0]==np.array(save_history[1]))))


# plot losses
plt.plot(clf.history['loss'])

# plot auc
plt.plot(clf.history['train_auc'])
plt.plot(clf.history['valid_auc'])

# plot learning rates
plt.plot(clf.history['lr'])



"""
Predictions
"""
preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")



"""
Global explainability : feat importance summing to 1
"""
clf.feature_importances_


from sklearn.calibration import calibration_curve

fop, mpv = calibration_curve(y_test, preds[:,1], n_bins=10, normalize=True)


