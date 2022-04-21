import json

import torch

from libcity.data.dataset.dataset_subclass.RiskseqDataset import dataprocess
from libcity.model.traffic_accident_prediction.Riskseq import Riskseq

data_dic = dataprocess()

batch_size = 30
train_epoch = 50
learning_rate = 0.01

f = open('./libcity/config/model/traffic_state_pred/Riskseq.json', 'r')
config = json.load(f)
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Riskseq(config,data_dic).to(device)
model = Riskseq(config, data_dic)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(train_epoch):
    optimizer.zero_grad()
    model.calculate_loss(batch_size).backward()
    optimizer.step()
