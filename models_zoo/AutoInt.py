from models_zoo.layers import MLP, FeaturesEmbedding, InteractingLayer, FeaturesLinear, Stacked_Gate_unit, Concat_MLP
from models_zoo.Basic import *
import torch.nn as nn


class AutoInt_model(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, att_layer_num, att_head_num, att_res, mlp_dims, dropout=0):
        super(AutoInt_model, self).__init__()
        device = 'cuda:0'
        self.field_dims = field_dims
        field_size = len(field_dims)
        self.embed_dim = embed_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear_model = FeaturesLinear(field_dims)
        if len(mlp_dims) and att_layer_num > 0:
            dnn_linear_in_feature = mlp_dims[-1] + field_size * embed_dim
        elif len(mlp_dims) > 0:
            dnn_linear_in_feature = mlp_dims[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_size * embed_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embed_dim, att_head_num, att_res) for _ in range(att_layer_num)])
        self.to(device)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x_2d = torch.reshape(embed_x, (embed_x.shape[0], -1))
        logit = self.linear_model(x)
        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        if len(self.mlp_dims) > 0 and self.att_layer_num > 0:
            deep_out = self.mlp(embed_x_2d)
            stack_out = torch.cat([att_output, deep_out], dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.mlp_dims) > 0:  # Only Deep
            deep_out = self.mlp(embed_x_2d)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:
            pass
        return torch.sigmoid(logit.squeeze(1))


class AutoInt_enhanced(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, att_layer_num, att_head_num, att_res, mlp_dims, stacked_num, concat_mlp, dropout=0):
        super(AutoInt_enhanced, self).__init__()
        device = 'cuda:0'
        self.field_dims = field_dims
        field_size= len(field_dims)
        self.field_size = field_size
        self.embed_dim = embed_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear_model = FeaturesLinear(field_dims)
        if len(mlp_dims) and att_layer_num > 0:
            dnn_linear_in_feature = mlp_dims[-1] + field_size * embed_dim * 2
        elif len(mlp_dims) > 0:
            dnn_linear_in_feature = mlp_dims[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_size * embed_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embed_dim, att_head_num, att_res) for _ in range(att_layer_num)])
        self.to(device)

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
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        logit = self.linear_model(x)
        att_input = embed_x_3d
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        if len(self.mlp_dims) > 0 and self.att_layer_num > 0:
            if self.concat:
                deep_out = self.concat_mlp(embed_x, embed_x)
            else:
                deep_out = self.mlp(embed_x)
            stack_out = torch.cat([att_output, deep_out], dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.mlp_dims) > 0:  # Only Deep
            deep_out = self.mlp(embed_x)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:
            pass
        return torch.sigmoid(logit.squeeze(1))


class AutoInt_super(BasicSuper):
    def __init__(self, args, field_dims, embed_dim, att_layer_num, att_head_num, att_res, mlp_dims, stacked_num, concat_mlp, dropout=0):
        super().__init__(args, field_dims, embed_dim)
        device = 'cuda:0'
        self.field_dims = field_dims
        field_size = len(field_dims)
        self.field_size = field_size
        self.embed_dim = embed_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear_model = FeaturesLinear(field_dims)
        if len(mlp_dims) and att_layer_num > 0:
            dnn_linear_in_feature = mlp_dims[-1] + field_size * embed_dim * 2
        elif len(mlp_dims) > 0:
            dnn_linear_in_feature = mlp_dims[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_size * embed_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embed_dim, att_head_num, att_res) for _ in range(att_layer_num)])
        self.to(device)

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
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, phase, mask_num)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        logit = self.linear_model(x)
        att_input = embed_opt_3d
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        if len(self.mlp_dims) > 0 and self.att_layer_num > 0:
            if self.concat:
                deep_out = self.concat_mlp(embed_opt_2d, embed_opt_2d)
            else:
                deep_out = self.mlp(embed_x)
            stack_out = torch.cat([att_output, deep_out], dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.mlp_dims) > 0:  # Only Deep
            deep_out = self.mlp(embed_opt_2d)
            logit = self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit = self.dnn_linear(att_output)
        else:
            pass
        return torch.sigmoid(logit.squeeze(1))


class AutoInt_evo(BasicEvo):
    def __init__(self, args, field_dims, embed_dim, att_layer_num, att_head_num, att_res, mlp_dims, stacked_num, concat_mlp, dropout=0):
        super().__init__(args, field_dims, embed_dim)
        device = 'cuda:0'
        self.field_dims = field_dims
        field_size = len(field_dims)
        self.field_size = field_size
        self.embed_dim = embed_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear_model = FeaturesLinear(field_dims)
        if len(mlp_dims) and att_layer_num > 0:
            dnn_linear_in_feature = mlp_dims[-1] + field_size * embed_dim * 2
        elif len(mlp_dims) > 0:
            dnn_linear_in_feature = mlp_dims[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_size * embed_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embed_dim, att_head_num, att_res) for _ in range(att_layer_num)])
        self.to(device)

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat = concat_mlp

        # # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x, cand):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d, cand)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        logit = self.linear_model(x)
        att_input = embed_opt_3d
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        if len(self.mlp_dims) > 0 and self.att_layer_num > 0:
            if self.concat:
                deep_out = self.concat_mlp(embed_opt_2d, embed_opt_2d)
            else:
                deep_out = self.mlp(embed_x)
            stack_out = torch.cat([att_output, deep_out], dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.mlp_dims) > 0:  # Only Deep
            deep_out = self.mlp(embed_opt_2d, embed_opt_2d)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:
            pass
        return torch.sigmoid(logit.squeeze(1))


class AutoInt_retrain(BasicRetrain):
    def __init__(self, args, field_dims, embed_dim, att_layer_num, att_head_num, att_res, mlp_dims, stacked_num, concat_mlp, dropout=0):
        super().__init__(args, field_dims, embed_dim)
        device = 'cuda:0'
        self.field_dims = field_dims
        field_size = len(field_dims)
        self.field_size = field_size
        self.embed_dim = embed_dim
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear_model = FeaturesLinear(field_dims)
        if len(mlp_dims) and att_layer_num > 0:
            dnn_linear_in_feature = mlp_dims[-1] + field_size * embed_dim * 2
        elif len(mlp_dims) > 0:
            dnn_linear_in_feature = mlp_dims[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_size * embed_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embed_dim, att_head_num, att_res) for _ in range(att_layer_num)])
        self.to(device)

        # # todo Stacked_Gate
        stacked_type = 'normal_bit'
        gate_type = 'normal_bit'
        stacked_num = stacked_num
        self.stacked_gate_unit = Stacked_Gate_unit(field_size, embed_dim, stacked_type, gate_type, stacked_num)
        self.concat_mlp = concat_mlp

        # todo Concat_MLP
        input_dim = field_size * embed_dim
        self.concat_mlp = Concat_MLP(field_size, embed_dim, input_dim, mlp_dims, dropout, output_layer=False)

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x, embed_weight = self.stacked_gate_unit(embed_x)

        embed_x_3d = torch.reshape(embed_x, (embed_x.shape[0], self.field_size, -1))
        embed_opt_3d = self.calculate_ele_input(embed_x_3d)
        embed_opt_2d = torch.reshape(embed_opt_3d, (embed_opt_3d.shape[0], -1))
        logit = self.linear_model(x)
        att_input = embed_opt_3d
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        if len(self.mlp_dims) > 0 and self.att_layer_num > 0:
            if self.concat_mlp:
                deep_out = self.concat_mlp(embed_opt_2d, embed_opt_2d)
            else:
                deep_out = self.mlp(embed_x)
            stack_out = torch.cat([att_output, deep_out], dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.mlp_dims) > 0:  # Only Deep
            # deep_out = self.mlp(embed_x)
            deep_out = self.concat_mlp(embed_opt_2d, embed_opt_2d)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:
            pass
        return torch.sigmoid(logit.squeeze(1))

