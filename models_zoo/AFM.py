from models_zoo.layers import AttentionalFactorizationMachine, FeaturesEmbedding, FeaturesLinear, Stacked_Gate_unit,Concat_MLP
from models_zoo.Basic import *

class AFM_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class AFM_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, attn_size, stacked_num, dropouts):
        super(AFM_enhanced, self).__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        x = self.linear(x) + self.afm(embed_x_3d)
        return torch.sigmoid(x.squeeze(1))


class AFM_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, attn_size, stacked_num, dropouts):
        super(AFM_super, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    def forward(self, x, phase, mask_num):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        x = self.linear(x) + self.afm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))


class AFM_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, attn_size, stacked_num, dropouts):
        super(AFM_evo, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    def forward(self, x, cand):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        x = self.linear(x) + self.afm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))


class AFM_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, attn_size, stacked_num, dropouts):
        super(AFM_retrain, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        x = self.linear(x) + self.afm(embed_opt_3d)
        return torch.sigmoid(x.squeeze(1))












