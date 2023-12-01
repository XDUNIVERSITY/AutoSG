from models_zoo.Basic import *
from models_zoo.layers import FeaturesLinear, MLP, FeaturesEmbedding, Stacked_Gate_unit, Concat_MLP


class WideDeep_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        embed_x = self.embedding(x)
        # x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x1 = self.linear(x)
        x2 = self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = x1 + x2
        return torch.sigmoid(x.squeeze(1))


class WideDeep_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        field_size = len(field_dims)

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        if self.concat:
            x = self.linear(x) + self.concat_mlp(embed_x, embed_x)
        else:
            x = self.linear(x) + self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))


class WideDeep_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        field_size = len(field_dims)
        self.field_size = field_size

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x, phase, mask_num):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        if self.concat:
            x = self.linear(x) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(embed_opt_2d)
        return torch.sigmoid(x.squeeze(1))


class WideDeep_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        field_size = len(field_dims)
        self.field_size = field_size

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x, cand):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        if self.concat:
            x = self.linear(x) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(embed_opt_2d)
        return torch.sigmoid(x.squeeze(1))


class WideDeep_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        field_size = len(field_dims)
        self.field_size = field_size

        # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        if self.concat:
            x = self.linear(x) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(embed_opt_2d)
        return torch.sigmoid(x.squeeze(1))
