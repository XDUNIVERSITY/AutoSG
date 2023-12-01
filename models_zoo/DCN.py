from models_zoo.layers import FeaturesEmbedding, MLP, CrossNetwork, Stacked_Gate_unit, Concat_MLP
from models_zoo.Basic import *


class DCN_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super(DCN_model, self).__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        field_size = len(field_dims)
        self.mlp_dims = mlp_dims
        self.field_size = field_size
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))


class DCN_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        field_size = len(field_dims)
        self.mlp_dims = mlp_dims
        self.field_size = field_size
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        self.field_size = field_size

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        x_l1 = self.cn(embed_x)
        if self.concat:
            h_l2 = self.concat_mlp(embed_x, embed_x)
        else:
            h_l2 = self.mlp(embed_x)
        # h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))


class DCN_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, num_layers, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        field_size = len(field_dims)
        self.mlp_dims = mlp_dims
        self.field_size = field_size
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        self.field_size = field_size
        self.lazylinear = torch.nn.LazyLinear(1)

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x, phase, mask_num):
        embed_x_raw = self.embedding(x).view(-1, self.embed_output_dim)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x_raw)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))

        x_l1 = self.cn(embed_opt_2d)
        if self.concat:
            h_l2 = self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            h_l2 = self.mlp(embed_opt_2d)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.lazylinear(x_stack)
        return torch.sigmoid(p.squeeze(1))


class DCN_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, num_layers, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        field_size = len(field_dims)
        self.mlp_dims = mlp_dims
        self.field_size = field_size
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        self.field_size = field_size
        self.lazylinear = torch.nn.LazyLinear(1)

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x, cand):
        embed_x_raw = self.embedding(x).view(-1, self.embed_output_dim)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x_raw)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        x_l1 = self.cn(embed_opt_2d)
        if self.concat:
            h_l2 = self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            h_l2 = self.mlp(embed_opt_2d)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.lazylinear(x_stack)
        return torch.sigmoid(p.squeeze(1))


class DCN_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, num_layers, mlp_dims, stacked_num, concat_mlp, dropout):
        super().__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        field_size = len(field_dims)
        self.mlp_dims = mlp_dims
        self.field_size = field_size
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        self.field_size = field_size
        self.lazylinear = torch.nn.LazyLinear(1)

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x):
        embed_x_raw = self.embedding(x).view(-1, self.embed_output_dim)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x_raw)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))

        x_l1 = self.cn(embed_opt_2d)
        if self.concat:
            h_l2 = self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            h_l2 = self.mlp(embed_opt_2d)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.lazylinear(x_stack)
        return torch.sigmoid(p.squeeze(1))

