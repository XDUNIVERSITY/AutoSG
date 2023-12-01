from models_zoo.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, Stacked_Gate_unit, MLP, Concat_MLP
from models_zoo.Basic import *


class NFM_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super(NFM_model, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MLP(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class NFM_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropouts):
        super(NFM_enhanced, self).__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_dim, mlp_dims, dropouts[1])
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat_mlp = concat_mlp

        # todo Concat_MLP
        input_dim = embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropouts[1], output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        cross_term = self.fm(embed_x_3d)
        if self.concat_mlp:
            x = self.linear(x) + self.mlp(cross_term)
        else:
            x = self.linear(x) + self.concat_mlp(cross_term, embed_x)
        return torch.sigmoid(x.squeeze(1))


class NFM_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropouts):
        super(NFM_super, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_dim, mlp_dims, dropouts[1])
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp
        # todo Concat_MLP
        input_dim = embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropouts[1], output_layer=True)

    def forward(self, x, phase, mask_num):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        cross_term = self.fm(embed_opt_3d)

        if self.concat:
            x = self.linear(x) + self.concat_mlp(cross_term, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class NFM_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropouts):
        super(NFM_evo, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_dim, mlp_dims, dropouts[1])
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp
        # todo Concat_MLP
        input_dim = embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropouts[1], output_layer=True)

    def forward(self, x, cand):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        cross_term = self.fm(embed_opt_3d)
        if self.concat:
            x = self.linear(x) + self.concat_mlp(cross_term, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))


class NFM_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, mlp_dims, stacked_num, concat_mlp, dropouts):
        super(NFM_retrain, self).__init__(args, field_dims, embed_dim)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.stacked_num = stacked_num
        self.dropout = dropouts
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_dim, mlp_dims, dropouts[1])
        field_size = len(field_dims)
        self.field_size = field_size
        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # todo Concat_MLP
        input_dim = embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropouts[1], output_layer=True)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)
        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        cross_term = self.fm(embed_opt_3d)
        if self.concat:
            x = self.linear(x) + self.concat_mlp(cross_term, embed_opt_2d)
        else:
            x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))

