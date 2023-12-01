import shutil
import struct
from collections import defaultdict
from pathlib import Path
import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm

class AvazuDataset(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    You can get avazu dataset from this website
        https://www.kaggle.com/c/avazu-ctr-prediction
    """
    def __init__(self, dataset_path=None, cache_path='./dataset/avazu/.avazu_whole', rebuild_cache=False, min_threshold=4):
        self.NUM_FEATS = 22
        self.num_sparse_feats = 22
        self.num_dense_feats = 0
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e9)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items(): # fm:{'255': 0, '91': 1, '33': 2, '112': 3, '93': 4, '13': 5, '90': 6, '126': 7, '251': 8, '101': 9, '108': 10, '163': 11, '178': 12, '156': 13, '70': 14, '194': 15, '104': 16, '79': 17, '116': 18, '212': 19, '94': 20, '182': 21, '48': 22, '117': 23, '177': 24, '111': 25, '159': 26, '102': 27, '42': 28, '69': 29, '17': 30, '15': 31, '16': 32, '68': 33, '110': 34, '157': 35, '32': 36, '61': 37, '100': 38, '76': 39, '195': 40, '35': 41, '219': 42, '46': 43, '204': 44, '43': 45, '82': 46, '1': 47, '171': 48, '51': 49, '52': 50, '221': 51, '253': 52, '229': 53, '95': 54, '23': 55, '20': 56, '246': 57, '71': 58}
                field_dims[i - 1] = len(fm) + 1 # field_dims: [ 241    8    8  923  735   19  667   57   19   89 4287 2135    5    5, 1397    9   10  386    5   64  150   60]
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                #todo
                # values[i] 代表每行的第i+1个值
                # 1. 构建feat_cnts字典存储每个特征域的所出现的特征值以及每个特征值出现的次数，feat_cntes [i][j] 代表第i个特征域对应的第j个值出现的个数, defaultdict(<class 'int'>,{3: {'1': 56245, '0': 145535, '5': 30, '2': 71, '4': 30, '7': 226, '3': 8}}
                # 2. 构建 feat_mapper字典去点出现次数少的，把每个特征域的特征值按顺序进行排列{3:{'5': 0, '4': 1, '1': 2, '0': 3, '7': 4, '3': 5, '2': 6}}
                # 3. 构建default字典，记录每个特征域特征值个数。{1: 240, 2: 7, 3: 7, 4: 922, 5: 734, 6: 18, 7: 666, 8: 56, 9: 18, 10: 88, 11: 4286, 12: 2134, 13: 4, 14: 4, 15: 1396, 16: 8, 17: 9, 18: 385, 19: 4, 20: 63, 21: 149, 22: 59}
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}# cnt: defaultdict(<class 'int'>, {'14102102': 1087, '14102104': 1299, '14102103': 967, '}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()} #
      #feat_mappper: { …… 2:{'1010': 0, '1002': 1, '1008': 2, '1012': 3, '1005': 4, '1001': 5, '1007': 6} 3：{'5': 0, '0': 1, '2': 2, '7': 3, '4': 4, '3': 5, '1': 6}

       # defaults{1: 240, 2: 7, 3: 7, 4: 922, 5: 734, 6: 18, 7: 666, 8: 56, 9: 18, 10: 88, 11: 4286, 12: 21332, 13: 4, 14: 4,
       #  15: 1396, 16: 8, 17: 9, 18: 385, 19: 4, 20: 63, 21: 149, 22: 59}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

