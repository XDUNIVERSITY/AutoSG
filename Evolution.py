import argparse
import collections
import os.path
import tqdm
import time
from dataset.split_data import *
from utilis.utils import *
from sklearn.metrics import roc_auc_score
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
# Please fill in the path of your dataset in the default option
parser.add_argument('--dataset_path', default=' ')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--seed', type=int, default=2018, help="random seed")
parser.add_argument('--save_path', default='save')
parser.add_argument('--embed_dim', default=16)

#todo load evo-name
parser.add_argument('--supernet_random_name', default='model_random_train.pth')

#todo save-arch-name
parser.add_argument('--search_random_arch', default='search_random_arch.pth')
parser.add_argument('--debug_mode', type=int, default=1)
parser.add_argument("--mode_supernet", type=str, default="random", help="random")

#todo supernet
parser.add_argument("--model_super_name", type=str, default="DeepFM_super", help="supernet_name")

#todo stacked_gate_para
parser.add_argument('--stacked_num', type=int, default=5, help="stacked_gate_layer")
parser.add_argument('--concat_mlp', type=bool, default=True, help="use concat_mlp")

parser.add_argument("--mask_num", type=int, default=5, help="mask number")
parser.add_argument("--keep_num", type=int, default=0, help="keep number")
parser.add_argument("--mutation_num", type=int, default=10, help="mutation number")
parser.add_argument("--crossover_num", type=int, default=10, help="crossover number")
parser.add_argument("--m_prob", type=float, default=0.1, help="Mutation Probatbility")
parser.add_argument("--norm", type=int, default=1, help="norm used")
parser.add_argument('--search_epoch', type=int, default=30)
args = parser.parse_args()

ROOT_PATH = os.path.abspath(os.path.join(__file__, '../'))
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_TIME = str(TIMESTAMP)


def get_model(name, dataset, logger):
    field_dims = dataset.field_dims
    stacked_num = args.stacked_num
    concat = args.concat_mlp
    logger.info("[Dataset :{dataset}]".format(dataset=args.dataset))
    logger.info("[Model :{model_name}]".
                format(model_name=name))
    logger.info("[Stacked_num :{stacked_num:d} | Concat:{concat}]".
                     format(stacked_num=stacked_num, concat=concat))
    logger.info("[Mask_number :{mask_num}]".
                format(mask_num=args.mask_num))
    logger.info("[mutation_num :{mutation_num} | crossover_num:{crossover_num} | m_prob:{m_prob}]".
                     format(mutation_num=args.mutation_num, crossover_num=args.crossover_num, m_prob=args.m_prob))
    if name == 'FM_evo':
        return FM_evo(args, field_dims, embed_dim=16, stacked_num=6)
    if name == 'DeepFM_evo':
        return DeepFM_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'NFM_evo':
        return NFM_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropouts=(0, 0))
    if name == 'AFM_evo':
        return AFM_evo(args, field_dims, embed_dim=16, attn_size=16, stacked_num=stacked_num, dropouts=(0, 0))
    if name == 'WideDeep_evo':
        return WideDeep_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'FNN_evo':
        return FNN_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'DCN_evo':
        return DCN_evo(args, field_dims, embed_dim=16, num_layers=3, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num, concat_mlp=concat,
                       dropout=0)
    if name == 'IPNN_evo':
        return PNN_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), stacked_num=stacked_num,
                       concat_mlp=concat, dropout=0)
    if name == 'AutoInt_evo':
        return AutoInt_evo(args, field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True, mlp_dims=(1024, 1024, 1024),
                           stacked_num=stacked_num, concat_mlp=concat, dropout=0)
    if name == 'XDeepFM_evo':
        return XDeepFM_evo(args, field_dims, embed_dim=16, mlp_dims=(1024, 1024, 1024), dropout=0, cross_layer_sizes=(100, 100, 100),
                           stacked_num=stacked_num, concat_mlp=concat, split_half=True)


class EvolutionSearcher(object):
    def __init__(self, args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader, field_dims, device):
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.test_loader = test_dataloader
        self.save_path = os.path.join(args.save_path, args.dataset, args.model_super_name,
                                      args.mode_supernet)
        self.debug_mode = args.debug_mode
        self.batch_size = args.batch_size
        self.field_num = len(field_dims)
        self.embed_dim = args.embed_dim
        self.embedding_size = self.field_num * self.embed_dim
        self.logger = set_logger(model_name, args.dataset, args.stacked_num, args.concat_mlp, TIMESTAMP, ROOT_PATH)
        self.model = get_model(model_name, dataset, self.logger).to(device)
        self.mode_supernet = args.mode_supernet
        # todo load evo model
        if self.mode_supernet == 'random':
            supernet_name = args.supernet_random_name + str(args.mask_num)
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, supernet_name)),
                                       strict=False)
        else:
            raise Exception(f"wrong load mode")

        # todo save_arch_name
        self.save_random_arch = args.search_random_arch
        self.criterion = torch.nn.BCELoss()

        # Evolutionary Search Hyper-params
        self.mask_num = args.mask_num
        self.population_num = args.keep_num + args.mutation_num + args.crossover_num
        self.keep_num = args.keep_num
        self.mutation_num = args.mutation_num
        self.crossover_num = args.crossover_num
        self.m_prob = args.m_prob
        # self.logger = set_logger(model_name, args.dataset, TIMESTAMP, ROOT_PATH)
        self.device = device

    def get_ele_random(self, num, mask_num):
        print("Generating random embedding masks ...")
        self.cands = []
        for i in range(num):
            cand = torch.randint(low=0, high=self.embedding_size, size=(mask_num,)).to(self.device)
            self.cands.append(cand)

    def __save_arch(self, cand):
        os.makedirs(self.save_path, exist_ok=True)
        embed_mask = cand
        save_dict = collections.OrderedDict([("embed_mask", embed_mask)])
        if self.mode_supernet == 'random':
            ranom_path = self.save_random_arch + str(self.mask_num)
            torch.save(save_dict, os.path.join(self.save_path, ranom_path))
        else:
            raise Exception(f"wrong save mode")

    def eval_one_part(self, model, data_loader, device, cand):
        model.eval()
        targets, predicts = list(), list()
        total_loss = 0
        batch_num = 0
        with torch.no_grad():
            for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(device), target.to(device)
                y = model(fields, cand)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                loss = self.criterion(y, target.float())
                total_loss += loss.item()
                batch_num += 1
            auc = roc_auc_score(targets, predicts)
            loss = total_loss/batch_num
        return auc, loss

    def eval_all_parts(self, model, dataloader, device):
        aucs, losses = [], []
        for i, cand in enumerate(self.cands):
            auc, loss = self.eval_one_part(model, dataloader, device, cand)
            aucs.append(auc)
            losses.append(loss)
        return aucs, losses

    def sort_cands(self, metrics):
        reverse = [1 - i for i in metrics]
        indexlist = np.argsort(reverse)
        self.cands = [self.cands[i] for i in indexlist]

    def get_ele_mutation(self, mutation_num, m_prob, mask_num):
        mutation = []
        assert m_prob > 0
        for i in range(mutation_num):
            origin = self.cands[i]
            for i in range(mask_num):
                if random.random() < m_prob:
                    index = torch.tensor(i).to(self.device)
                    rand_value = torch.randint(low=0, high=self.embedding_size, size=(1,)).to(self.device)
                    origin[index] = rand_value
            mutation.append(origin)
        return mutation

    def get_ele_crossover(self, crossover_num, mask_num):
        crossover = []
        def indexes_gen(m, n):
            seen = set()
            x, y = random.randint(m, n), random.randint(m, n)
            while True:
                seen.add((x, y))
                yield (x, y)
                x, y = random.randint(m, n), random.randint(m, n)
                while (x, y) in seen:
                    x, y = random.randint(m, n), random.randint(m, n)
        gen = indexes_gen(0, crossover_num)
        for i in range(crossover_num):
            point = random.randint(1, mask_num)
            x, y = next(gen)
            origin_x, origin_y = self.cands[x].cpu().numpy(), self.cands[y].cpu().numpy()
            xy = np.concatenate((origin_x[:point], origin_y[point:]))
            crossover.append(torch.from_numpy(xy).to(self.device))
        return crossover

    def ele_search(self, max_epoch):
        self.logger.info('-' * 80)
        self.logger.info('Begin Searching ...')
        self.get_ele_random(self.population_num, args.mask_num)
        acc_best_auc = 0.0
        acc_cand = []
        for epoch_idx in range(int(max_epoch)):
            aucs, losses = self.eval_all_parts(self.model, self.valid_loader, self.device)
            self.logger.info("Epoch = {} | best AUC {} | worst AUC {}".format(epoch_idx, max(aucs), min(aucs)))
            self.sort_cands(aucs)
            self.sort_cands(losses)
            if acc_best_auc < aucs[0]:
                acc_best_auc, acc_cand = aucs[0], self.cands[0]

            mutation = self.get_ele_mutation(self.mutation_num, self.m_prob, args.mask_num)
            crossover = self.get_ele_crossover(self.crossover_num, args.mask_num)
            self.cands = self.cands[:self.keep_num] + mutation + crossover
            acc_test_auc, acc_test_logloss = self.eval_one_part(self.model, self.test_loader, self.device,
                                                                cand=acc_cand)
            self.logger.info(
                "Test Accurate | AUC: {} | Logloss: {}".format(acc_test_auc, acc_test_logloss))

            self.logger.info("Accurate Cand: {}".format(acc_cand))
            if self.debug_mode == 1:
                self.__save_arch(acc_cand)
                self.logger.info("Model saved")


def main(args):
    dataset, train_dataloader, valid_dataloader, test_dataloader = get_split_data(args.dataset, args.dataset_path,
                                                                                  args.batch_size)
    model_name = 'DeepFM_evo'
    searcher = EvolutionSearcher(args, model_name, dataset, train_dataloader, valid_dataloader, test_dataloader,
                                 dataset.field_dims, device='cuda:0')
    searcher.ele_search(args.search_epoch)


if __name__ == '__main__':
    set_seed(args.seed)
    main(args)

