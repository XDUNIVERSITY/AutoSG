from dataset.split_data import *
from utilis.utils import *
from utilis.train_test import *
from models_zoo.FM import *
from models_zoo.DeepFM import *
from models_zoo.NFM import *
from models_zoo.AFM import *
from models_zoo.WideDeep import *
from models_zoo.FNN import *
from models_zoo.DCN import *
from models_zoo.IPNN import *
from models_zoo.AutoInt import *
from models_zoo.XDeepFM import *
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe',
                    choices=['criteo', 'avazu', 'movielens', 'frappe'])
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--seed', type=int, default=2018, help="random seed")
parser.add_argument('--use_gpu', type=bool, default=True)
# Please fill in the path of your dataset in the default option
parser.add_argument('--dataset_path', default=' ')
parser.add_argument('--save_path', default='save')
parser.add_argument('--init_name', default='model_init.pth')
parser.add_argument('--debug_mode', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--save_dir', default='model_save')
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-5)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--mask_num", type=int, default=5, help="mask number")

#todo save
parser.add_argument('--save_random_name', default='model_random_train.pth')

#todo stacked_gate_para
parser.add_argument('--stacked_num', type=int, default=5, help="stacked_gate_layer")
parser.add_argument('--concat_mlp', type=bool, default=True, help="use concat_mlp")

#todo mode
parser.add_argument("--mode_supernet", type=str, default="random", help="random")
parser.add_argument("--norm", type=int, default=1)
parser.add_argument("--alpha", type=float, default=3e-4)
parser.add_argument('--train_epoch', type=int, default=30)
args = parser.parse_args()
ROOT_PATH = os.path.abspath(os.path.join(__file__, '../'))
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_TIME = str(TIMESTAMP)


def get_model(name, dataset, logger):
    field_dims = dataset.field_dims
    stacked_num = args.stacked_num
    concat = args.concat_mlp
    logger.info("[Dataset :{dataset}]".format(dataset=args.dataset))
    logger.info("[Model :{model_name}]".format(model_name=name))
    logger.info("[Stacked_num :{stacked_num:d} | Concat:{concat}]".format(stacked_num=stacked_num, concat=concat))
    logger.info("[Mask_number :{mask_num}]".format(mask_num=args.mask_num))
    if name == 'FM_super':
        return FM_super(args, field_dims, embed_dim=16, stacked_num=stacked_num)
    if name == 'DeepFM_super':
        return DeepFM_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                            concat_mlp=concat, dropout=0)
    if name == 'NFM_super':
        return NFM_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                         concat_mlp=concat, dropouts=(0, 0))
    if name == 'AFM_super':
        return AFM_super(args, field_dims, embed_dim=16, attn_size=16, stacked_num=stacked_num, dropouts=(0, 0))
    if name == 'WideDeep_super':
        return WideDeep_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                              concat_mlp=concat, dropout=0)
    if name == 'FNN_super':
        return FNN_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                         concat_mlp=concat, dropout=0)
    if name == 'DCN_super':
        return DCN_super(args, field_dims, embed_dim=16, num_layers=3, mlp_dims=(1024, 1024, 1024),
                         stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'IPNN_super':
        return PNN_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                         concat_mlp=concat, dropout=0)
    if name == 'AutoInt_super':
        return AutoInt_super(args, field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True, mlp_dims=(1024, 1024, 1024),
                             stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'XDeepFM_super':
        return XDeepFM_super(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0, cross_layer_sizes=(100, 100, 100),
                             stacked_num=stacked_num, concat_mlp=concat, split_half=True)


class SuperNet(object):  
    def __init__(self, args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader, device):
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.test_loader = test_dataloader
        self.save_path = os.path.join(args.save_path, args.dataset, model_name,
                                      args.mode_supernet)
        self.save_random_name = args.save_random_name
        self.init_name = args.init_name
        self.debug_mode = args.debug_mode
        self.batch_size = args.batch_size
        self.alpha = args.alpha
        self.norm = args.norm
        self.logger = set_logger(model_name, args.dataset, args.stacked_num, args.concat_mlp, TIMESTAMP, ROOT_PATH)
        self.model = get_model(model_name, dataset, self.logger).to(device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.device = device
        self.mode_supernet = args.mode_supernet

    def __save_model(self, save_name):
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        torch.save(self.model.state_dict(), os.path.join(dir_path, self.save_path, save_name))

    def train_epoch_random(self, model, optimizer, data_loader, criterion, device):
        model.train()
        total_loss = 0
        batch_num = 0
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        phase = 'random_train'
        for i, (fields, target) in enumerate(tk0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields, phase, args.mask_num)
            logloss = criterion(y, target.float())
            loss = logloss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        train_loss = total_loss/batch_num
        return train_loss

    def train_epoch(self, mode):
        if mode == 'random':
            train_loss = self.train_epoch_random(self.model, self.optimizer, self.train_loader,
                                                 self.criterion, self.device)
        else:
            raise Exception(f"wrong train mode")
        return train_loss

    def get_save_path(self, mode):
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        if mode == 'random':
            self.save_random_name = self.save_random_name + str(args.mask_num)
            save_path = os.path.join(dir_path, self.save_path, self.save_random_name)
        else:
            raise Exception(f"wrong save mode")
        return save_path

    def eval_one_part(self, model, data_loader, device):
        model.eval()
        targets, predicts = list(), list()
        phase = 'test'
        total_loss = 0
        step = 0
        with torch.no_grad():
            for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(device), target.to(device)
                y = model(fields, phase, args.mask_num)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                loss = self.criterion(y, target.float())
                total_loss += loss
                step += 1
            auc = roc_auc_score(targets, predicts)
            loss = total_loss/step
        return auc, loss

    def train_test(self, max_epoch):
        print('-' * 80)
        print('Begin Training ...')
        if self.debug_mode == 1:
            self.__save_model(self.init_name)
        save_path = self.get_save_path(self.mode_supernet)
        early_stopper = EarlyStopper(num_trials=3, save_path=save_path)
        auc_list = []
        loss_list = []
        for epoch_idx in range(int(max_epoch)):
            train_loss = self.train_epoch(self.mode_supernet)
            val_auc, val_loss = self.eval_one_part(self.model, self.valid_loader, self.device)
            test_auc, test_loss = self.eval_one_part(self.model, self.test_loader, self.device)
            self.logger.info("[Epoch {epoch:d} | Train Loss: {loss:.6f}]".
                                 format(epoch=epoch_idx, loss=train_loss))
            self.logger.info("[Epoch {epoch:d} | Val Loss:{loss:.6f} | Val AUC: {auc:.6f}]".
                             format(epoch=epoch_idx, loss=val_loss, auc=val_auc))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f}| Test AUC: {auc:.6f}]".
                             format(epoch=epoch_idx, loss=test_loss, auc=test_auc))
            if not early_stopper.is_continue(self.model, test_auc, test_loss):
                auc_list.append(early_stopper.best_accuracy)
                loss_list.append(early_stopper.loss)
                self.logger.info(auc_list)
                break
        self.logger.info("Most Accurate | AUC: {} | Logloss: {}".format(early_stopper.best_accuracy, early_stopper.loss))
        return early_stopper.best_accuracy


def main(args, times):
    dataset, train_dataloader, valid_dataloader, test_dataloader = get_split_data(args.dataset, args.dataset_path,
                                                                               args.batch_size)
    auc_list = []
    for i in range(times):
        model_name = 'DeepFM_super'
        sner = SuperNet(args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader, device='cuda:0')
        auc = sner.train_test(args.train_epoch)
        auc_list.append(auc)
    print(auc_list)
    print('average:')
    print(get_average(auc_list))


if __name__ == '__main__':
    set_seed(args.seed)
    main(args, times=1)


