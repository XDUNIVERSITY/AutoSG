from models_zoo.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, Stacked_Gate_unit
from models_zoo.Basic import *


class FM_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class FM_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, stacked_num):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    # x = [batchsize * 22]
    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], -1, self.embed_dim))
        x = self.linear(x) + self.fm(embed_x_3d)
        return torch.sigmoid(x.squeeze(1))


class FM_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, stacked_num):
        super(FM_super, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    # x = [batchsize * 22]
    def forward(self, x, phase, mask_num):
        embed = self.embedding(x)
        embed_x, emb_weight = self.stacked_gate_unit(embed)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        x = self.linear(x) + self.fm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))


class FM_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, stacked_num):
        super(FM_evo, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    # x = [batchsize * 22]
    def forward(self, x, cand):
        embed = self.embedding(x)
        embed_x, emb_weight = self.stacked_gate_unit(embed)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        x = self.linear(x) + self.fm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))


class FM_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, stacked_num):
        super(FM_retrain, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    # x = [batchsize * 22]
    def forward(self, x):
        embed = self.embedding(x)
        embed_x, emb_weight = self.stacked_gate_unit(embed)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        x = self.linear(x) + self.fm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))
