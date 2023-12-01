import argparse
import time

from utilis.train_test import *
from sklearn.metrics import roc_auc_score
from utilis.utils import *
from dataset.split_data import *
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe',
                    choices=['criteo', 'avazu', 'movielens', 'frappe'])
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--seed', type=int, default=2018, help="random seed")
parser.add_argument('--use_gpu', type=bool, default=True)
# Please fill in the path of your dataset in the default option
parser.add_argument('--dataset_path', default=' ')
parser.add_argument('--save_path', default='./save')
parser.add_argument('--arch_name', default='best_arch.pth')
parser.add_argument('--save_name', default='retrain.pth')
parser.add_argument('--init_name', default='model_init.pth')
parser.add_argument('--random_weight_name', default='model_random_train.pth')
parser.add_argument('--random_arch_name', default='search_random_arch.pth')

# todo notice here to change the base model
parser.add_argument("--model_super_name", type=str, default="DeepFM_super", help="supernet_name")

parser.add_argument('--debug_mode', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--save_dir', default='model_save')
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-5)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")

# todo main mode
parser.add_argument("--mode_main", type=str, default="run_original", help="run_ssg, run_original, retrain")
parser.add_argument("--mode_retrain", type=str, default="random", help="random")


#todo stacked_gate_para
parser.add_argument('--stacked_num', type=int, default=8, help="stacked_gate_layer")
parser.add_argument('--concat_mlp', type=bool, default=True, help="use concat_mlp")


#todo mask_para
parser.add_argument('--mask_num', type=int, default=5, help="mask_num")


parser.add_argument("--norm", type=int, default=1)
parser.add_argument("--alpha", type=float, default=3e-4)
parser.add_argument('--train_epoch', type=int, default=30)
parser.add_argument('--compare_with_MSG', type=bool, default=True)
args = parser.parse_args()

ROOT_PATH = os.path.abspath(os.path.join(__file__, '../'))
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_TIME = str(TIMESTAMP)


def get_original_model(name, dataset, logger):
    field_dims = dataset.field_dims
    logger.info("[Dataset :{dataset}]".format(dataset=args.dataset))
    logger.info("[Original Model :{model_name}]".
                format(model_name=name))
    if name == 'FM':
        return FM_model(field_dims, embed_dim=16)
    if name == 'DeepFM':
        return DeepFM_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0)
    if name == 'NFM':
        return NFM_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropouts=(0, 0))
    if name == 'AFM':
        return AFM_model(field_dims, embed_dim=16, attn_size=16, dropouts=(0, 0))
    if name == 'WideDeep':
        return WideDeep_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0)
    if name == 'FNN':
        return FNN_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0)
    if name == 'DCN':
        return DCN_model(field_dims, embed_dim=16, num_layers=3, mlp_dims=(1024, 1024, 1024), dropout=0)
    if name == 'IPNN':
        return PNN_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0)
    if name == 'AutoInt':
        return AutoInt_model(field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True,
                             mlp_dims=(512, 256, 128), dropout=0)
    if name == 'XDeepFM':
        return XDeepFM_model(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0,
                             cross_layer_sizes=(100, 100, 100), split_half=True)


def get_enhanced_model(name, dataset, logger):
    field_dims = dataset.field_dims
    stacked_num = args.stacked_num
    concat = args.concat_mlp
    logger.info("[Dataset :{dataset}]".format(dataset=args.dataset))
    logger.info("[Model :{model_name}]".
                format(model_name=name))
    logger.info("[Stacked_num :{stacked_num:d} | Concat:{concat}]".
                     format(stacked_num=stacked_num, concat=concat))
    if name == 'FM_enhanced':
        return FM_enhanced(field_dims, embed_dim=16, stacked_num=6)
    if name == 'DeepFM_enhanced':
        return DeepFM_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                               concat_mlp=concat, dropout=0)
    if name == 'NFM_enhanced':
        return NFM_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                            concat_mlp=concat, dropouts=(0, 0))
    if name == 'AFM_enhanced':
        return AFM_enhanced(field_dims, embed_dim=16, attn_size=16, stacked_num=stacked_num, dropouts=(0, 0))
    if name == 'WideDeep_enhanced':
        return WideDeep_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                                 concat_mlp=concat, dropout=0)
    if name == 'FNN_enhanced':
        return FNN_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                            concat_mlp=concat, dropout=0)
    if name == 'DCN_enhanced':
        return DCN_enhanced(field_dims, embed_dim=16, num_layers=3, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                            concat_mlp=concat, dropout=0)
    if name == 'IPNN_enhanced':
        return PNN_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                            concat_mlp=concat, dropout=0)
    if name == 'AutoInt_enhanced':
        return AutoInt_enhanced(field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True, mlp_dims=(1024, 1024, 1024),
                                stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'XDeepFM_enhanced':
        return XDeepFM_enhanced(field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0, cross_layer_sizes=(100, 100, 100),
                                stacked_num=stacked_num,  concat_mlp=concat, split_half=True)


def retrain_model(name, dataset, logger):
    field_dims = dataset.field_dims
    stacked_num = args.stacked_num
    concat = args.concat_mlp
    logger.info("[Dataset :{dataset}]".format(dataset=args.dataset))
    logger.info("[Model :{model_name}]".
                format(model_name=name))
    logger.info("[Stacked_num :{stacked_num:d} | Concat:{concat}]".
                     format(stacked_num=stacked_num, concat=concat))
    logger.info("[Mask_num :{mask_num:d}]".
                format(mask_num=args.mask_num))
    if name == 'FM_retrain':
        return FM_retrain(args, field_dims, embed_dim=16, stacked_num=6)
    if name == 'DeepFM_retrain':
        return DeepFM_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'NFM_retrain':
        return NFM_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropouts=(0, 0))
    if name == 'AFM_retrain':
        return AFM_retrain(args, field_dims, embed_dim=16, attn_size=16, stacked_num=stacked_num, dropouts=(0, 0))
    if name == 'WideDeep_retrain':
        return WideDeep_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'FNN_retrain':
        return FNN_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'DCN_retrain':
        return DCN_retrain(args, field_dims, embed_dim=16, num_layers=3, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat,
                           dropout=0)
    if name == 'IPNN_retrain':
        return PNN_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'AutoInt_retrain':
        return AutoInt_retrain(args, field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True,
                               mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'XDeepFM_retrain':
        return XDeepFM_retrain(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0,
                               cross_layer_sizes=(100, 100, 100), stacked_num=stacked_num, concat_mlp=concat, split_half=True)


class Retrainer(object):
    def __init__(self, args, model_name, dataset, train_dataloader, valid_dataloader,
                 test_dataloader, device):
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.test_loader = test_dataloader
        self.model_name = model_name
        self.save_path = os.path.join(args.save_path, args.dataset, args.model_super_name,
                                          args.mode_retrain)
        self.save_name = args.save_name
        self.init_name = args.init_name
        self.debug_mode = args.debug_mode
        self.alpha = args.alpha
        self.norm = args.norm
        self.criterion = torch.nn.BCELoss()
        self.device = device
        self.mode_main = args.mode_main
        self.mode_retrain = args.mode_retrain
        self.mask_num = args.mask_num
        self.random_weight_name = args.random_weight_name
        self.random_arch_name = args.random_arch_name
        self.logger = set_logger(model_name, args.dataset, args.stacked_num, args.concat_mlp, TIMESTAMP, ROOT_PATH)

        if self.mode_main == 'run_ssg':
            self.model = get_enhanced_model(model_name, dataset, self.logger).to(device)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_params}")
            if args.compare_with_MSG:
                init = torch.load(os.path.join(self.save_path, self.init_name))
                self.model.load_state_dict(init, strict=False)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        elif self.mode_main == 'run_original':
            self.model = get_original_model(model_name, dataset, self.logger).to(device)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_params}")
            if args.compare_with_MSG:
                init = torch.load(os.path.join(self.save_path, self.init_name))
                self.model.load_state_dict(init, strict=False)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        elif self.mode_main == 'retrain':
            self.model = retrain_model(model_name, dataset, self.logger).to(device)
            if self.init_name != 'init.pth':
                init = torch.load(os.path.join(self.save_path, self.init_name))
                self.model.load_state_dict(init, strict=False)
            if self.mode_retrain == 'random':
                random_path = self.random_arch_name + str(self.mask_num)
                arch = torch.load(os.path.join(self.save_path, random_path))
                self.model.update_ele_mask(embed_ele_mask=arch['embed_mask'])
            else:
                raise Exception("wrong retrain mode")
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        else:
            raise Exception("wrong main mode")

    def __save_model(self, save_name):
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        torch.save(self.model.state_dict(), os.path.join(dir_path, self.save_path, save_name))

    def train_epoch(self, model, optimizer, data_loader, criterion, device, log_interval=100):
        model.train()
        total_loss = 0
        batch_num = 0
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, (fields, target) in enumerate(tk0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
        return total_loss / batch_num

    def eval_one_part(self, model, data_loader, device):
        model.eval()
        targets, predicts = list(), list()
        total_loss = 0
        batch_num = 0
        with torch.no_grad():
            for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(device), target.to(device)
                y = model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                loss = self.criterion(y, target.float())
                total_loss += loss
                batch_num += 1
            auc = roc_auc_score(targets, predicts)
            loss = total_loss / batch_num
        return auc, loss

    def train_test(self, max_epoch):
        print('-' * 80)
        print('Begin Training ...')
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        save_path = os.path.join(dir_path, self.save_path, self.save_name)
        early_stopper = EarlyStopper(num_trials=3, save_path=save_path)
        auc_list = []
        loss_list = []
        for epoch_idx in range(int(max_epoch)):
            train_loss = self.train_epoch(self.model, self.optimizer, self.train_loader,
                                          self.criterion, self.device)
            test_auc, test_loss = self.eval_one_part(self.model, self.test_loader, self.device)
            self.logger.info("[Epoch {epoch:d} | Train Loss: {loss:.6f}]".
                             format(epoch=epoch_idx, loss=train_loss))
            self.logger.info("[Epoch {epoch:d} | Test Loss:{loss:.6f} | Test AUC: {auc:.6f}]".
                             format(epoch=epoch_idx, loss=test_loss, auc=test_auc))
            if not early_stopper.is_continue(self.model, test_auc, test_loss):
                auc_list.append(early_stopper.best_accuracy)
                loss_list.append(early_stopper.loss)
                self.logger.info(auc_list)
                break
        self.logger.info(
            "Most Accurate | AUC: {} | Logloss: {}".format(early_stopper.best_accuracy, early_stopper.loss))
        return early_stopper.best_accuracy, early_stopper.loss


def main(args, times):
    set_seed(args.seed)
    dataset, train_dataloader, valid_dataloader, test_dataloader = get_split_data(args.dataset, args.dataset_path,
                                                                                  args.batch_size)
    auc_list = []
    loss_list = []
    for i in range(times):
        if args.mode_main == 'run_ssg':
            model_name = 'DeepFM_enhanced'
            rter = Retrainer(args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader,
                             device='cuda:0')
            auc, loss = rter.train_test(args.train_epoch)
            auc_list.append(auc)
            loss_list.append(loss)
        elif args.mode_main == 'run_original':
            model_name = 'DeepFM'
            rter = Retrainer(args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader,
                             device='cuda:0')
            auc, loss = rter.train_test(args.train_epoch)
            auc_list.append(auc)
            loss_list.append(loss)
        elif args.mode_main == 'retrain':
            model_name = 'DeepFM_retrain'
            rter = Retrainer(args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader,
                             device='cuda:0')
            auc, loss = rter.train_test(args.train_epoch)
            auc_list.append(auc)
            loss_list.append(loss)
        else:
            raise Exception('wrong main mode')
    logger_result(times, auc_list, loss_list, model_name, rter.logger)


if __name__ == '__main__':
    set_seed(args.seed)
    main(args, times=1)
