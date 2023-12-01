from models_zoo.layers import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, MLP, Stacked_Gate_unit, Concat_MLP
from models_zoo.Basic import *


class XDeepFM_model(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):

        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        # x = self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class XDeepFM_enhanced(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, stacked_num, concat_mlp, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)
        field_size = len(field_dims)
        self.field_size = field_size
        self.concat = concat_mlp

        ## todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)

        ## todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))

        if self.concat:
            x = self.linear(x) + self.cin(embed_x_3d) + self.concat_mlp(embed_x, embed_x)
        else:
            x = self.linear(x) + self.cin(embed_x_3d) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(x.squeeze(1))


class XDeepFM_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, stacked_num, concat_mlp, split_half=True):
        super(XDeepFM_super, self).__init__(args, field_dims, embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
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
            x = self.linear(x) + self.cin(embed_opt_3d) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(x.squeeze(1))


class XDeepFM_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, stacked_num, concat_mlp, split_half=True):
        super(XDeepFM_evo, self).__init__(args, field_dims, embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)
        field_size = len(field_dims)
        self.field_size = field_size

        # # todo Stacked_Gate
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
            x = self.linear(x) + self.cin(embed_opt_3d) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class XDeepFM_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, stacked_num, concat_mlp, split_half=True):
        super(XDeepFM_retrain, self).__init__(args, field_dims, embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)
        field_size = len(field_dims)
        self.field_size = field_size

        # # todo Stacked_Gate
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
            x = self.linear(x) + self.cin(embed_opt_3d) + self.concat_mlp(embed_opt_2d, embed_opt_2d)
        else:
            x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

