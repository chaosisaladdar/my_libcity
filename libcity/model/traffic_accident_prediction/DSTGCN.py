from logging import getLogger
from typing import List

import dgl
import torch
import torch.nn as nn
from dgl import init as g_init
from dgl.nn.pytorch import GraphConv

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.loss import masked_mae_loss


class fully_connected_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize fullyConnectedNet.

        Parameters
        ----------
        input_size – The number of expected features in the input x  -> scalar
        hidden_size – The numbers of features in the hidden layer h  -> list
        output_size  – The number of expected features in the output x  -> scalar

        input -> (batch, in_features)

        :return
        output -> (batch, out_features)

        """
        super(fully_connected_layer, self).__init__()

        self.input_size = input_size
        # list
        self.hidden_size = hidden_size
        self.output_size = output_size
        fcList = []
        reluList = []
        for index in range(len(self.hidden_size)):
            if index != 0:
                input_size = self.hidden_size[index - 1]
            fc = nn.Linear(input_size, self.hidden_size[index])
            setattr(self, f'fc{index}', fc)
            fcList.append(fc)
            relu = nn.ReLU()
            setattr(self, f'relu{index}', relu)
            reluList.append(relu)
        self.last_fc = nn.Linear(self.hidden_size[-1], self.output_size)

        self.fcList = nn.ModuleList(fcList)
        self.reluList = nn.ModuleList(reluList)

    def forward(self, input_tensor):

        """
        :param input_tensor:
            2-D Tensor  (batch, input_size)

        :return:
            2-D Tensor (batch, output_size)
            output_tensor
        """
        for idx in range(len(self.fcList)):
            out = self.fcList[idx](input_tensor)
            out = self.reluList[idx](out)
            input_tensor = out
        # (batch, output_size)
        output_tensor = self.last_fc(input_tensor)

        return output_tensor


# spatial_layer.py
class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for idx, hidden_size in enumerate(hidden_sizes):
            if idx == 0:
                gcns.append(GraphConv(in_features, hidden_size))
                relus.append(nn.ReLU())
                bns.append(nn.BatchNorm1d(hidden_size))
            else:
                gcns.append(GraphConv(hidden_sizes[idx - 1], hidden_size))
                relus.append(nn.ReLU())
                bns.append(nn.BatchNorm1d(hidden_size))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        gcns.append(GraphConv(hidden_sizes[-1], out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor):
        """
        :param g: a graph
        :param node_features: shape (n_nodes, n_features)
        :return:
        """
        g.set_n_initializer(g_init.zero_initializer)
        g.set_e_initializer(g_init.zero_initializer)
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            h = gcn(g, h)
            if len(h.shape) > 2:
                h = bn(h.transpose(1, -1)).transpose(1, -1)
            else:
                h = bn(h)
            h = relu(h)
        return h


# spatial_layer.py too
class StackedSBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, h = input
        for module in self[:-1]:
            h = h + module(g, h)
        h = self[-1](g, h)
        return h


# spatial_temporal_layer.py
class STBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        """
        :param f_in: the number of dynamic features each node before
        :param f_out: the number of dynamic features each node after
        """
        super(STBlock, self).__init__()
        # stack four middle layers to transform features from f_in to f_out
        self.spatial_embedding = GCN(f_in, [(f_in * (4 - i) + f_out * i) // 4 for i in (1, 4)], f_out)
        self.temporal_embedding = nn.Conv1d(f_out, f_out, 3, padding=1)

    def forward(self, g: dgl.DGLGraph, temporal_features: torch.Tensor):
        """
        :param g: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param temporal_features: shape [node_num, f_in, t_in]
        :return: hidden features after temporal and spatial embedding, shape [node_num, f_out, t_in]
        """
        return self.temporal_embedding(self.spatial_embedding(g, temporal_features.transpose(-2, -1)).transpose(-2, -1))


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSTBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, h = input
        for module in self:
            h = torch.cat((h, module(g, h)), dim=1)

        return h


class _DSTGCN(nn.Module):
    def __init__(self, f_1: int, f_2: int, f_3: int):
        """
        :param f_1: the number of spatial features each node, default 22
        :param f_2: the number of dynamic features each node, default 1
        :param f_3: the number of features overall (external)
        """
        super(_DSTGCN, self).__init__()

        self.spatial_embedding = fully_connected_layer(f_1, [20], 15)
        self.spatial_gcn = StackedSBlocks([GCN(15, [15, 15, 15], 15),
                                           GCN(15, [15, 15, 15], 15),
                                           GCN(15, [14, 13, 12, 11], 10)])
        self.temporal_embedding = StackedSTBlocks([STBlock(f_2, 4), STBlock(5, 5), STBlock(10, 10)])

        self.temporal_agg = nn.AvgPool1d(24)

        self.external_embedding = fully_connected_layer(f_3, [(f_3 * (4 - i) + 10 * i) // 4 for i in (1, 4)], 10)

        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(10 + 20 + 10, 1),
                                          nn.Sigmoid())

    def forward(self,
                bg: dgl.DGLGraph,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                external_features: torch.Tensor):
        """
        get predictions
        :param bg: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param spatial_features: shape [node_num, F_1]
        :param temporal_features: shape [node_num, F_2, T]
        :param external_features: shape [batch_size, F_3]
        :return: a tensor, shape [batch_size], with the prediction results for each graphs
        """

        s_out = self.spatial_gcn(bg, self.spatial_embedding(spatial_features))

        temporal_embeddings = self.temporal_embedding(bg, temporal_features)

        # t_out of shape [1, node_num, 10]
        t_out = self.temporal_agg(temporal_embeddings)
        t_out.squeeze_()

        e_out = self.external_embedding(external_features)

        nums_nodes, id = bg.batch_num_nodes(), 0
        s_features, t_features = list(), list()
        for num_nodes in nums_nodes:
            s_features.append(s_out[id])
            t_features.append(t_out[id])
            id += num_nodes

        s_features = torch.stack(s_features)
        t_features = torch.stack(t_features)

        output_features = torch.cat((s_features[0], t_features[0], e_out), -1)

        return self.output_layer(output_features)


class DSTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """
        # 1.初始化父类（必须）
        super(DSTGCN, self).__init__(config, data_feature)
        # 2.从data_feature获取想要的信息，注意不同模型使用不同的Dataset类，其返回的data_feature内容不同（必须）
        self.device = config.get('device', 'cpu')
        self._scaler = self.data_feature.get('scaler')
        # self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        # 以TrafficStateGridDataset为例演示取数据，可以取出如下的数据，用不到的可以不取
        # **这些参数的不能从config中取的**
        self.f_1 = self.data_feature.get("f_1", 22)
        self.f_2 = self.data_feature.get("f_2", 1)
        self.f_3 = self.data_feature.get("f_3", 43)
        # 3.初始化log用于必要的输出（必须）
        self._logger = getLogger()
        # 4.初始化device（必须）
        self.device = config.get('device', torch.device('cpu'))
        # 5.初始化输入输出时间步的长度（非必须）
        # self.input_window = config.get('input_window', 1)
        # self.output_window = config.get('output_window', 1)
        # 6.从config中取用到的其他参数，主要是用于构造模型结构的参数（必须）
        # 这些涉及到模型结构的参数应该放在trafficdl/config/model/model_name.json中（必须）
        # 例如: self.blocks = config['blocks']
        # ...
        # 7.构造深度模型的层次结构（必须）
        # 例如: 使用简单RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.dstgcn = _DSTGCN(f_1=self.f_1, f_2=self.f_2, f_3=self.f_3)

    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return:
        """
        # 1.取数据，假设字典中有4类数据，X,y,X_ext,y_ext
        # 当然一般只需要取输入数据，例如X,X_ext，因为这个函数是用来计算输出的
        # 模型输入的数据的特征维度应该等于self.feature_dim
        # x = batch['X']  # shape = (batch_size, input_length, ..., feature_dim)
        # 例如: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        g = batch['g']
        bg = g[0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bg = bg.to(device)
        spatial_features = batch['spatial_features']
        s_f = spatial_features[0]
        temporal_features = batch['temporal_features']
        t_f = temporal_features[0]
        external_features = batch['external_features']
        e_f = external_features[0]
        # 2.根据输入数据计算模型的输出结果
        # 模型输出的结果的特征维度应该等于self.output_dim
        # 模型输出的结果的其他维度应该跟batch['y']一致，只有特征维度可能不同（因为batch['y']可能包含一些外部特征）
        # 如果模型的单步预测，batch['y']是多步的数据，则时间维度也有可能不同
        # 例如: outputs = self.model(x)
        outputs = self.dstgcn.forward(bg, s_f, t_f, e_f)
        # 3.返回输出结果
        # 例如: return outputs
        return outputs

    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程这个batch数据的loss，也就是需要定义一个loss函数。
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: training loss (tensor)
        """
        # 1.取出真值 ground_truth
        # y_true = batch['y']
        y_true = batch['y']
        y_true = y_true[0]
        # 2.取出预测值
        y_predicted = self.predict(batch)
        # 3.使用self._scaler将进行了归一化的真值和预测值进行反向归一化（必须）
        # y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4.调用loss函数计算真值和预测值的误差
        # trafficdl/model/loss.py中定义了常见的loss函数
        # 如果模型源码用到了其中的loss，则可以直接调用，以MSE为例:
        loss = masked_mae_loss(y_predicted, y_true)
        # 如果模型源码所用的loss函数在loss.py中没有，则需要自己实现loss函数
        # ...（自定义loss函数）
        # 5.返回loss的结果
        return loss

    def predict(self, batch):
        """
        输入一个batch的数据，返回对应的预测值，一般应该是**多步预测**的结果
        一般会调用上边定义的forward()方法
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: predict result of this batch (tensor)
        """
        # 如果self.forward()的结果满足要求，可以直接返回
        # 如果不符合要求，例如self.forward()进行了单时间步的预测，但是模型训练时使用的是每个batch的数据进行的多步预测，
        # 则可以参考trafficdl/model/traffic_speed_prediction/STGCN.py中的predict()函数，进行多步预测
        # 多步预测的原则是: 先进行一步预测，用一步预测的结果进行二步预测，**而不是使用一步预测的真值进行二步预测!**
        # 以self.forward()的结果符合要求为例:
        return self.forward(batch)
