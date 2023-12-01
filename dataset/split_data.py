import torch
from dataset.avazu.avazu import AvazuDataset
from dataset.criteo.criteo import CriteoDataset
from dataset.frappe.frappe import FrappeDataset
from dataset.movielens.movies import MovieLens20MDataset
from torch.utils.data import DataLoader


def get_dataset(name, path):
    global cache_path
    if name == 'avazu':
        if name == 'avazu':
            cache_path = './dataset/avazu/.avazu_whole'
        return AvazuDataset(path, cache_path)
    if name == 'criteo':
        if name == 'criteo':
            cache_path = './dataset/criteo/.criteo_whole'
        return CriteoDataset(path, cache_path)
    if name == 'movielens':
        cache_path = './dataset/Movielens/.movielens_whole'
        return MovieLens20MDataset(path, cache_path)
    if name == 'frappe':
        cache_path = './dataset/Frappe/.frappe_whole'
        return FrappeDataset(path, cache_path)


def get_split_data(dataset_name, dataset_path, batch_size):
    # train : valid : test = 8 : 1 : 1
    if dataset_name == 'avazu':
        dataset = get_dataset(dataset_name, dataset_path)
    elif dataset_name == 'criteo':
        dataset = get_dataset(dataset_name, dataset_path)
    elif dataset_name == 'frappe':
        dataset = get_dataset(dataset_name, dataset_path)
    elif dataset_name == 'movielens':
        dataset = get_dataset(dataset_name, dataset_path)
    else:
        raise Exception(f"No such dataset {dataset_name}!")
    print(len(dataset))
    # avazu: 40428967, train:32343172 , valid:4,042,897, test: 4,042,897
    if dataset_name == 'avazu':
        train_length = 32343173
        valid_length = 4042897
        test_length = 4042897

    # criteo: 45,840,617, train:36,672,493 , valid:4,584,062 , test: 4,584,062
    elif dataset_name == 'criteo':
        train_length = 36672493
        valid_length = 4584062
        test_length = 4584062

    # frappe: 288609, train: 230887 , valid:28861, valid:28861
    elif dataset_name == 'frappe':
        train_length = 230887
        valid_length = 28861
        test_length = 28861

    # movielens-20M:2006859, train:1605487, valid:200686, test:200686
    elif dataset_name == 'movielens':
        train_length = 1605487
        valid_length = 200686
        test_length = 200686
    else:
        raise Exception('NO dataset')

    if dataset_name == 'movielens':
        train_dataset = torch.utils.data.Subset(dataset, range(train_length))
        valid_dataset = torch.utils.data.Subset(dataset, range(train_length, train_length + valid_length))
        test_dataset = torch.utils.data.Subset(dataset, range(train_length + valid_length, train_length + valid_length + test_length))
    else:
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    return dataset, train_data_loader, valid_data_loader, test_data_loader
