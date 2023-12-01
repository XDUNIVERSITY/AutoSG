import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,), requires_grad=True))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        fc = self.fc(x)
        wx = torch.sum(fc, dim=1)
        lr = wx + self.bias
        return lr


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims[:-1])), dtype=np.long)
        torch.nn.init.xavier_normal(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class Gate_Module(torch.nn.Module):
    def __init__(self, gate_type, embedding_size, field_size):
        super(Gate_Module, self).__init__()
        self.raw_dim = embedding_size * field_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.gate_type = gate_type

        if gate_type == 'normal_bit':
            # nn.init.constant_(self.w, 1)
            self.sofmax = nn.Softmax(dim=1)
            gate_list = list()
            self.hidden_size = self.raw_dim
            self.fs_layer1 = gate_list.append(nn.Linear(self.raw_dim, self.hidden_size))
            self.activate1 = gate_list.append(torch.nn.ReLU())
            # self.activate1 = gate_list.append(torch.nn.Tanh())
            self.fs_layer2 = gate_list.append(torch.nn.Linear(self.hidden_size, self.raw_dim))
            self.activate2 = gate_list.append(torch.nn.Sigmoid())
            self.gate = torch.nn.Sequential(*gate_list)

        if gate_type == 'normal_vector':
            # nn.init.constant_(self.w, 1)
            self.sofmax = nn.Softmax(dim=1)
            gate_list = list()
            # self.hidden_size = 128
            self.hidden_size = self.raw_dim
            self.fs_layer1 = gate_list.append(torch.nn.Linear(self.raw_dim, self.hidden_size))
            self.activate1 = gate_list.append(torch.nn.ReLU())
            self.fs_layer2 = gate_list.append(torch.nn.Linear(self.hidden_size, self.field_size))
            self.activate2 = gate_list.append(torch.nn.Sigmoid())
            self.gate = torch.nn.Sequential(*gate_list)
        self.batchnorm = torch.nn.LayerNorm(normalized_shape=[self.embedding_size])

    def forward(self, x):
        emb = x
        emb_concat = torch.reshape(emb, (emb.shape[0], -1))
        if self.gate_type == 'normal_bit':
            a_weight = self.gate(emb_concat)
            feature = a_weight * emb_concat
            if len(x.shape) == 3:
                feature = torch.reshape(feature, (feature.shape[0], -1, self.embedding_size))
            # return feature
            return feature, a_weight

        if self.gate_type == 'normal_vector':
            if len(x.shape) == 2:
                emb = torch.reshape(x, (x.shape[0], -1, self.embedding_size))
            a_weight = self.gate(emb_concat)
            a_combine = torch.unsqueeze(a_weight, dim=2)
            a_repeat = a_combine.repeat(1, 1, self.embedding_size)
            feature = a_repeat * emb
            if len(x.shape) == 2:
                feature = torch.reshape(x, (x.shape[0], -1))
            return feature, a_weight


class Stacked_Gate_unit(torch.nn.Module):
    def __init__(self, field_size, embedding_size, stacked_type, gate_type, staked_num):
        super().__init__()
        self.field_size = field_size
        self.embed_dim = embedding_size
        self.embed_output_dim = field_size * embedding_size
        self.stacked_num = staked_num
        self.stacked_type = stacked_type
        self.gate_module = Gate_Module(stacked_type, embedding_size, field_size)
        self.gate_type = gate_type
        self.gate_bit = Gate_Module(gate_type, embedding_size, field_size)
        self._w_bit = torch.nn.Parameter(torch.empty(self.embed_output_dim))
        torch.nn.init.ones_(self._w_bit)

    # input:  embed_x = [batchsize, field x embedding_size]
    # output: embed_x = [batchsize, field x embedding_size]
    def forward(self, x):
        embed = x
        embed_x = torch.reshape(embed, (embed.shape[0], -1))
        emb_list = []
        gate_list = []
        weight_accum = 1
        for i in range(self.stacked_num):
            last_emb = embed_x
            emb_list.append(last_emb)
            gate_emb, a_weight = self.gate_module(embed_x)
            gate_list.append(gate_emb)
            new_emb = last_emb + gate_emb
            embed_x = new_emb
            weight_accum = weight_accum * (1 + a_weight)
        emb_bit, a_weight1 = self.gate_bit(embed_x)
        gate_list.append(emb_bit)
        w_bit = self._w_bit
        embed_x = w_bit * emb_bit
        emb_weight = w_bit * a_weight1 * weight_accum
        return embed_x, emb_weight


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for output_dim in output_dims:
            layers.append(torch.nn.Linear(input_dim, output_dim))
            layers.append(torch.nn.BatchNorm1d(output_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = output_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Concat_MLP(nn.Module):
    def __init__(self, filed_size, embedding_size, input_dim, output_dims, dropout, output_layer=True):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_size = embedding_size
        self.raw_dim = filed_size * embedding_size
        self.num_layer = len(output_dims)
        output_dims = list(output_dims)
        self.output_dims_list = output_dims
        input_dims = [input_dim]
        for i in range(len(output_dims)):
            input_dims.append(output_dims[i] + self.raw_dim)
        self.input_dims = input_dims
        self.last_raw_dim = output_dims[-1]

        self.layer_list = nn.ModuleList()
        self.BatchNorm_list = nn.ModuleList()
        self.relu_list = nn.ModuleList()
        self.drop_list = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(input_dims[i], output_dims[i]))
            self.BatchNorm_list.append(torch.nn.BatchNorm1d(output_dims[i]))
            self.relu_list.append(nn.ReLU())
            self.drop_list.append(nn.Dropout(p=dropout))
        self.is_output = output_layer
        self.output_linear = nn.Linear(input_dims[-1], 1)

    # x: input units to the mlp layer
    # emb: embedding
    def forward(self, x, emb):
        input_unit = torch.reshape(x, (x.shape[0], -1))
        emb_concat = torch.reshape(emb, (emb.shape[0], -1))
        merge = []
        mlp_list = []
        merge.append(emb_concat)
        for i in range(self.num_layer):
            if i == 0:
                layer_out = self.layer_list[i](input_unit)
                norm_out = self.BatchNorm_list[i](layer_out)
                relu_out = self.relu_list[i](norm_out)
                drop_out = self.drop_list[i](relu_out)
                mlp_unit = drop_out
                mlp_list.append(mlp_unit)
            if i > 0:
                layer_input = torch.cat([mlp_list[i - 1], emb_concat], dim=1)
                layer_out = self.layer_list[i](layer_input)
                norm_out = self.BatchNorm_list[i](layer_out)
                relu_out = self.relu_list[i](norm_out)
                drop_out = self.drop_list[i](relu_out)
                mlp_unit = drop_out
                mlp_list.append(mlp_unit)
        if self.is_output:
            last_input = torch.cat([mlp_list[-1], emb_concat], dim=1)
            output = self.output_linear(last_input)
        else:
            output = torch.cat([mlp_list[-1], emb_concat], dim=1)
        return output


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class IPNN(torch.nn.Module):
    def forward(self, x):
        y_final = []
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                y = x[:, i:i + 1, :] * x[:, j:j + 1, :]
                y_sum = torch.sum(y, dim=2)
                y_final.append(y_sum)
        return torch.cat(y_final, dim=1)


class CrossInteraction(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteraction, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interact_out = self.weight(X_i) * X_0 + self.bias
        return interact_out


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0])
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1])
        return self.fc(attn_output)


class InteractingLayer(nn.Module):
    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=2018):
        super(InteractingLayer, self).__init__()
        device = 'cuda:0'
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))
        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)
        return result


class CompressedInteractionNetwork(torch.nn.Module):
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))