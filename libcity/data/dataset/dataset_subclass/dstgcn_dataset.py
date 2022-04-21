import csv
import math
import os
import sys

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch

from libcity.data.dataset import TrafficStateDataset
from libcity.data.utils import generate_dataloader_pad
# scaler
from libcity.utils import MinMax01Scaler

# 以下为复用
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


def ValidDate(date):
    # print(type(date))
    # print(date)
    # date = time.strftime(date, "%Y-%m-%d %H:%M:%S")
    if len(date) < 11:
        return date + 'T' + '00:00:00' + 'Z'
    date = date.split()
    date = date[0] + 'T' + date[1] + 'Z'
    # time.strptime(date, "%Y/%m/%dT%H:%M:%SZ")
    return date


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def convert_by_type(lng, lat, type):
    if type == 'g2w':
        return gcj02_to_wgs84(lng, lat)
    else:
        print('Usage: type must be in one of g2b, b2g, w2g, g2w, b2w, w2b')
        sys.exit()


longitudeMin = 116.09608
longitudeMax = 116.71040
latitudeMin = 39.69086
latitudeMax = 40.17647

longitudeMin, latitudeMin = convert_by_type(lng=longitudeMin, lat=latitudeMin, type="g2w")
longitudeMax, latitudeMax = convert_by_type(lng=longitudeMax, lat=latitudeMax, type="g2w")

# 1110m divide
divideBound = 5

# grid divide
widthSingle = 0.01 / math.cos(latitudeMin / 180 * math.pi) / divideBound
width = math.floor((longitudeMax - longitudeMin) / widthSingle)
heightSingle = 0.01 / divideBound
height = math.floor((latitudeMax - latitudeMin) / heightSingle)


class DSTGCNDataset(TrafficStateDataset):
    """
    交通状态预测数据集的基类。
    默认使用`input_window`的数据预测`output_window`对应的数据，即一个X，一个y。
    一般将外部数据融合到X中共同进行预测，因此数据为[X, y]。
    默认使用`train_rate`和`eval_rate`在样本数量(num_samples)维度上直接切分训练集、测试集、验证集。
    """

    def __init__(self, config):
        super(DSTGCNDataset, self).__init__(config)
        # 我怀疑是这个东西不停读一堆图进去把内存在炸了，要是炸了我就把里面默认的给删了，跑个两跳的数据
        # 这里也暂时改一下
        # self.k_order = self.config.get('K_hop', 2)
        self.k_order = 2
        self.sf_mean = np.array(self.config.get('spatial_features_mean', 0))
        self.sf_std = np.array(self.config.get('spatial_features_std', 0))
        self.tf_mean = np.array(self.config.get('temporal_features_mean', 0))
        self.tf_std = np.array(self.config.get('temporal_features_std', 0))
        self.ef_mean = np.array(self.config.get('external_features_mean', 0))
        self.ef_std = np.array(self.config.get('external_features_std', 0))
        # 目测加载不了缓存，那就先摸一下
        self.cache_dataset = False
        # self.batch_size = self.config.get('batch_size', 1)
        # 这个地方改一下
        self.batch_size = 1
        self.grid_len_row = self.config.get('grid_len_row', 242)
        self.grid_len_column = self.config.get('grid_len_column', 236)
        print('******************************')
        print(self.data_path)
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):  # .rel file is not necessary
            self._load_rel()
        else:
            raise ValueError('Not found .rel file!')
        # 按继承基类，放到后面载入
        if os.path.exists(self.data_path + self.dataset + '.dyna'):
            self._load_dyna(self.dataset)
        else:
            raise ValueError('Not found .dyna file!')
        if os.path.exists(self.data_path + self.dataset + '.grid'):
            self._load_grid_3d(self.dataset)
        else:
            raise ValueError('Not found .grid file!')
        if os.path.exists(self.data_path + self.ext_file + '.ext'):
            self._load_ext()
        else:
            raise ValueError('Not found .ext file!')
        self.feature_name = {'g': 'no_tensor', 'spatial_features': 'no_pad_float', 'temporal_features': 'no_pad_float',
                             'external_features': 'no_pad_float', 'y': 'no_pad_float'}

    def _load_geo(self):
        #  载入节点
        geo_file = open(self.data_path + self.geo_file + '.geo', encoding='utf-8')
        geo_data = {'XCoord': [], 'YCoord': [], 'spatial_features': []}
        line = geo_file.readline()
        while True:
            line = geo_file.readline()
            if not line:
                break
            line = line.split(",", 6)
            # 用eval读入数字，否则读入字符串出大问题
            x_c = eval(line[2].lstrip('"['))
            y_c = eval(line[3].rstrip(']"'))
            s_f = eval(line[6].lstrip('"').rstrip('"\n'))
            geo_data['XCoord'].append(x_c)
            geo_data['YCoord'].append(y_c)
            geo_data['spatial_features'].append(s_f)
        geo_file.close()
        self.nodes = pd.DataFrame(geo_data)
        # 不知道会不会有问题
        # self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.nodes)))

    def _load_rel(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        加载.rel文件，格式[rel_id, type, origin_id, destination_id
        .rel文件用来表示路网关联数据

        Returns:
            self.rd_nwk: networkx.MultiDiGraph
        """
        # init road network, which is the result of this function
        self.network = nx.DiGraph(name="road_network")
        # load geo and rel file
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        # relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        # 得到节点数
        geo_num = geofile.shape[0]
        node_list = [i for i in range(geo_num)]
        edge_list = []
        with open(self.data_path + self.rel_file + '.rel', encoding='utf-8') as f:
            read_edges = csv.reader(f)
            headers = next(read_edges)
            for edge in read_edges:
                origin_id, destination_id = edge[2], edge[3]
                edge_list.append((int(origin_id), int(destination_id)))
        self.network.add_nodes_from(node_list)
        self.network.add_edges_from(edge_list)

        # logger
        # self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(geo_num))
        # self._logger.info("Loaded file " + self.rel_file + '.rel, num_roads=' + str(len(self.rd_nwk)))

    def _load_dyna(self, filename):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)，子类必须实现这个方法来指定如何加载数据文件，返回对应的多维数据,
        提供5个实现好的方法加载上述几类文件，并转换成不同形状的数组:
        `_load_dyna_3d`/`_load_grid_3d`/`_load_grid_4d`/`_load_grid_od_4d`/`_load_grid_od_6d`

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组
        """
        self._logger.info("Loading file " + filename + '.dyna')
        # 这个地方先加载测试版的数据
        self.accident = pd.read_csv(self.data_path + filename + '_test2.dyna')
        self.accident.drop(['dyna_id', 'type'], axis=1, inplace=True)
        self.sample_num = len(self.accident)

    def _load_grid_3d(self, filename):
        # self._logger.info("Loading file " + filename + '.grid')
        # grid_file = open(self.data_path + filename + '.grid', encoding='utf-8')
        # speed_data = {}
        time_index = pd.date_range(start="20180801000000", end="20181031230000", freq="H")  # freq="D"表示频率为每一天
        # for i in range(self.grid_len_row + 1):
        #     for j in range(self.grid_len_column + 1):
        #         speed_data[str(i) + ',' + str(j)] = []
        # line = grid_file.readline()
        # while True:
        #     line = grid_file.readline()
        #     if not line:
        #         break
        #     line = line.split(",")
        #     speed_data[line[3] + ',' + line[4]].append(eval(line[5].rstrip("\n")))
        # grid_file.close()
        # self.speed = pd.DataFrame(speed_data, time_index)
        # 测试阶段方便起见，用下面代码读入速度
        # 不可行，直接读csv会出现时间戳错误
        # 还是有问题, 还是识别不了
        speed = pd.read_csv(self.data_path + 'speed.csv', index_col=0)
        # speed.iloc[0, 0] = "date"
        # speed.drop(['date'], axis=1, inplace=True)
        speed_data = speed.to_dict(orient='list')
        self.speed = pd.DataFrame(speed_data, time_index)

    def _load_ext(self):
        """
        加载.ext文件，格式[ext_id, time, properties(若干列)],
        其中全局参数`ext_col`用于指定需要加载的数据的列，不设置则默认全部加载
        Returns:
            np.ndarray: 外部数据数组，shape: (timeslots, ext_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + self.ext_file + '.ext')
        ext_file = open(self.data_path + self.ext_file + '.ext', encoding='utf-8')
        ext_columns = ['temp', 'dewPt', 'rh', 'pressure', 'wspd', 'feels_like', 'wx_phrase_Cloudy',
                       'wx_phrase_Fair', 'wx_phrase_Fog', 'wx_phrase_Haze', 'wx_phrase_Heavy T-Storm',
                       'wx_phrase_Light Rain',
                       'wx_phrase_Light Rain Shower', 'wx_phrase_Light Rain with Thunder', 'wx_phrase_Mist',
                       'wx_phrase_Mostly Cloudy', 'wx_phrase_Partly Cloudy', 'wx_phrase_Rain', 'wx_phrase_T-Storm',
                       'wx_phrase_Thunder', 'wdir_cardinal_CALM', 'wdir_cardinal_E', 'wdir_cardinal_ENE',
                       'wdir_cardinal_ESE',
                       'wdir_cardinal_N', 'wdir_cardinal_NE', 'wdir_cardinal_NNE', 'wdir_cardinal_NNW',
                       'wdir_cardinal_NW',
                       'wdir_cardinal_S', 'wdir_cardinal_SE', 'wdir_cardinal_SSE', 'wdir_cardinal_SSW',
                       'wdir_cardinal_SW',
                       'wdir_cardinal_VAR', 'wdir_cardinal_W', 'wdir_cardinal_WNW', 'wdir_cardinal_WSW']
        weather_data = {'temp': [], 'dewPt': [], 'rh': [], 'pressure': [], 'wspd': [], 'feels_like': [],
                        'wx_phrase_Cloudy': [],
                        'wx_phrase_Fair': [], 'wx_phrase_Fog': [], 'wx_phrase_Haze': [], 'wx_phrase_Heavy T-Storm': [],
                        'wx_phrase_Light Rain': [],
                        'wx_phrase_Light Rain Shower': [], 'wx_phrase_Light Rain with Thunder': [],
                        'wx_phrase_Mist': [],
                        'wx_phrase_Mostly Cloudy': [], 'wx_phrase_Partly Cloudy': [], 'wx_phrase_Rain': [],
                        'wx_phrase_T-Storm': [],
                        'wx_phrase_Thunder': [], 'wdir_cardinal_CALM': [], 'wdir_cardinal_E': [],
                        'wdir_cardinal_ENE': [], 'wdir_cardinal_ESE': [],
                        'wdir_cardinal_N': [], 'wdir_cardinal_NE': [], 'wdir_cardinal_NNE': [], 'wdir_cardinal_NNW': [],
                        'wdir_cardinal_NW': [],
                        'wdir_cardinal_S': [], 'wdir_cardinal_SE': [], 'wdir_cardinal_SSE': [], 'wdir_cardinal_SSW': [],
                        'wdir_cardinal_SW': [],
                        'wdir_cardinal_VAR': [], 'wdir_cardinal_W': [], 'wdir_cardinal_WNW': [],
                        'wdir_cardinal_WSW': []}

        time_index = pd.date_range(start="20180801000000", end="20181031230000", freq="H")  # freq="D"表示频率为每一天
        line = ext_file.readline()
        while True:
            line = ext_file.readline()
            if line == '':
                break
            line = line.split(",")
            for i in range(len(ext_columns)):
                try:
                    weather_data[ext_columns[i]].append(eval(line[i + 2]))
                except:
                    print(i, line)
                    break
        ext_file.close()
        self.weather = pd.DataFrame(weather_data, time_index)

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以 g, spatial_features, temporal_features, external_features, target的形式返回

        Returns:
             g, spatial_features, temporal_features, external_features, target
        """
        sf_scaler = (self.sf_mean, self.sf_std)
        tf_scaler = (self.tf_mean, self.tf_std)
        ef_scaler = (self.ef_mean, self.ef_std)
        g_list, s_f_list, t_f_list, e_f_list, target_list = [], [], [], [], []
        for sample_id in range(self.sample_num):
            accident_time, node_id, target = self.accident.iloc[sample_id]
            accident_time = pd.Timestamp(accident_time)
            # get neighbors
            neighbors = nx.single_source_shortest_path_length(self.network, node_id, cutoff=self.k_order)
            # neighbors -> list
            neighbors.pop(node_id, None)
            neighbors = [node_id] + sorted(neighbors.keys())
            # get subgraph
            sub_graph = nx.subgraph(self.network, neighbors)
            sub_graph = nx.relabel_nodes(sub_graph, dict(zip(neighbors, range(len(neighbors)))))
            sub_graph.add_edges_from([(v, v) for v in sub_graph.nodes])
            g = dgl.DGLGraph(sub_graph)

            # get temporal_features (speed)
            # 这个地方可能存在潜在问题
            date_range = pd.date_range(end=accident_time.strftime("%Y%m%d %H"), freq="1H", periods=24)
            # print(date_range)
            selected_time = self.speed.loc[date_range]

            selected_nodes = self.nodes.loc[neighbors]

            spatial_features = selected_nodes['spatial_features'].tolist()

            x_ids = np.floor((selected_nodes['XCoord'].values - longitudeMin) / widthSingle).astype(np.int)
            y_ids = np.floor((selected_nodes['YCoord'].values - latitudeMin) / heightSingle).astype(np.int)

            temporal_features = selected_time[
                map(lambda ids: f'{ids[0]},{ids[1]}', zip(y_ids, x_ids))].values.transpose()

            # get external_features (weather + calendar)
            weather = self.weather.loc[date_range[-1]].tolist()
            external_features = weather + [accident_time.month, accident_time.day, accident_time.dayofweek,
                                           accident_time.hour, int(accident_time.dayofweek >= 5)]

            if sf_scaler is not None:
                mean, std = sf_scaler
                spatial_features = (np.array(spatial_features) - mean) / std
            if tf_scaler is not None:
                mean, std = tf_scaler
                temporal_features = (np.array(temporal_features) - mean) / std
            if ef_scaler is not None:
                mean, std = ef_scaler
                external_features = (np.array(external_features) - mean) / std

            # [N, F_1]
            spatial_features = torch.tensor(spatial_features).float()
            # [N, F_2, T]
            temporal_features = torch.tensor(temporal_features).unsqueeze(1).float()
            # [F_3]
            external_features = torch.tensor(external_features).float()

            target = torch.tensor(target).float()
            g_list.append(g)
            s_f_list.append(spatial_features)
            t_f_list.append(temporal_features)
            e_f_list.append(external_features)
            target_list.append(target)
        self._logger.info("Dataset created")
        return g_list, s_f_list, t_f_list, e_f_list, target_list

    def _split_train_val_test(self, g_list, s_f_list, t_f_list, e_f_list, target_list):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            输入列表形式的 g_list, s_f_list, t_f_list, e_f_list, target_list
            g, spatial_features, temporal_features, external_features, target
        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.sample_num * test_rate)
        num_train = round(self.sample_num * self.train_rate)
        num_val = self.sample_num - num_test - num_train

        # train
        g_train, s_f_train, t_f_train, e_f_train, target_train = g_list[:num_train], s_f_list[:num_train], t_f_list[
                                                                                                           :num_train], e_f_list[
                                                                                                                        :num_train], target_list[
                                                                                                                                     :num_train]
        # val
        g_val, s_f_val, t_f_val, e_f_val, target_val = g_list[num_train: num_train + num_val], s_f_list[
                                                                                               num_train: num_train + num_val], t_f_list[
                                                                                                                                num_train: num_train + num_val], e_f_list[
                                                                                                                                                                 num_train: num_train + num_val], target_list[
                                                                                                                                                                                                  num_train: num_train + num_val]
        # test
        g_test, s_f_test, t_f_test, e_f_test, target_test = g_list[-num_test:], s_f_list[-num_test:], t_f_list[
                                                                                                      -num_test:], e_f_list[
                                                                                                                   -num_test:], target_list[
                                                                                                                                -num_test:]

        return g_train, s_f_train, t_f_train, e_f_train, target_train, g_val, s_f_val, t_f_val, e_f_val, target_val, g_test, s_f_test, t_f_test, e_f_test, target_test

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        g_list, s_f_list, t_f_list, e_f_list, target_list = self._generate_data()
        return self._split_train_val_test(g_list, s_f_list, t_f_list, e_f_list, target_list)

    # 干脆就写一个get_data完事了
    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据
        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        g_train, s_f_train, t_f_train, e_f_train, target_train, g_val, s_f_val, t_f_val, e_f_val, target_val, g_test, s_f_test, t_f_test, e_f_test, target_test = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        g_train, s_f_train, t_f_train, e_f_train, target_train, g_val, s_f_val, t_f_val, e_f_val, target_val, g_test, s_f_test, t_f_test, e_f_test, target_test = self._generate_train_val_test()
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(g_train, s_f_train, t_f_train, e_f_train, target_train))
        eval_data = list(zip(g_val, s_f_val, t_f_val, e_f_val, target_val))
        test_data = list(zip(g_test, s_f_test, t_f_test, e_f_test, target_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader_pad(train_data, eval_data, test_data, self.feature_name,
                                    self.batch_size, self.num_workers)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征
        Returns:
            dict: 包含数据集的相关特征的字典
        """
        # raise NotImplementedError('Please implement the function `get_data_feature()`.')
        self.scaler = MinMax01Scaler(maxx=1, minn=0)
        self._logger.info('MinMax01Scaler max: ' + str(self.scaler.max) + ', min: ' + str(self.scaler.min))
        d = {"f_1": 22, "f_2": 1, "f_3": 43, "scaler": self.scaler, "output_dim": self.output_dim}
        return d
