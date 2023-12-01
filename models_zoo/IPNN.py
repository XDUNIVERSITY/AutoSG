from models_zoo.layers import FeaturesEmbedding, FeaturesLinear, IPNN, MLP, Stacked_Gate_unit, Concat_MLP
from models_zoo.Basic import *


class PNN_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.method = method
        field_size = len(field_dims)
        self.field_size = field_size
        self.pn = IPNN()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = field_size * embed_dim
        self.mlp = MLP(field_size * (field_size - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        cross_term = self.pn(embed_x)
        x = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(x.squeeze(1))


class PNN_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout, method='inner'):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.method = method
        field_size = len(field_dims)
        self.field_size = field_size
        self.pn = IPNN()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = field_size * embed_dim
        self.mlp = MLP(field_size * (field_size - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(self.field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # # todo Concat_MLP
        input_dim = self.field_size * (self.field_size - 1) // 2 + self.embed_output_dim
        self.concat_mlp = Concat_MLP(self.field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        cross_term = self.pn(embed_x)
        x_concat = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        if self.concat:
            x = self.concat_mlp(x_concat, embed_x)
        else:
            x = self.mlp(x_concat)
        return torch.sigmoid(x.squeeze(1))


class PNN_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout, method='inner'):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.method = method
        field_size = len(field_dims)
        self.field_size = field_size
        self.pn = IPNN()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = field_size * embed_dim
        self.mlp = MLP(field_size * (field_size - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(self.field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = self.field_size * (self.field_size - 1) // 2 + self.embed_output_dim
        self.concat_mlp = Concat_MLP(self.field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x, phase, mask_num):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))

        cross_term = self.pn(embed_opt_3d)
        x_concat = torch.cat([embed_opt_3d.view(-1, self.embed_output_dim), cross_term], dim=1)
        if self.concat:
            x = self.concat_mlp(x_concat, embed_opt_2d)
        else:
            x = self.mlp(x_concat)
        return torch.sigmoid(x.squeeze(1))


class PNN_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout, method='inner'):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.method = method
        field_size = len(field_dims)
        self.field_size = field_size
        self.pn = IPNN()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = field_size * embed_dim
        self.mlp = MLP(field_size * (field_size - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(self.field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = self.field_size * (self.field_size - 1) // 2 + self.embed_output_dim
        self.concat_mlp = Concat_MLP(self.field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x, cand):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))

        cross_term = self.pn(embed_opt_3d)
        x_concat = torch.cat([embed_opt_3d.view(-1, self.embed_output_dim), cross_term], dim=1)
        if self.concat:
            x = self.concat_mlp(x_concat, embed_opt_2d)
        else:
            x = self.mlp(x_concat)
        return torch.sigmoid(x.squeeze(1))


class PNN_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout, method='inner'):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.method = method
        field_size = len(field_dims)
        self.field_size = field_size
        self.pn = IPNN()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims, embed_dim)
        self.embed_output_dim = field_size * embed_dim
        self.mlp = MLP(field_size * (field_size - 1) // 2 + self.embed_output_dim, mlp_dims, dropout)

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(self.field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = self.field_size * (self.field_size - 1) // 2 + self.embed_output_dim
        self.concat_mlp = Concat_MLP(self.field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        cross_term = self.pn(embed_opt_3d)
        x_concat = torch.cat([embed_opt_3d.view(-1, self.embed_output_dim), cross_term], dim=1)
        if self.concat:
            x = self.concat_mlp(x_concat, embed_opt_2d)
        else:
            x = self.mlp(x_concat)
        return torch.sigmoid(x.squeeze(1))

