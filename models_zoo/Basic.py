import torch


class BasicModel(torch.nn.Module):
    def __init__(self, args, field_dims, embed_dim):
        super(BasicModel, self).__init__()
        self.device = torch.device("cuda:0")
        self.embed_dim = embed_dim
        self.field_dim = field_dims
        self.field_num = len(field_dims)
        self.feature_num = sum(field_dims)
        self.norm = args.norm
        self.embedding_size = self.field_num * self.embed_dim
        self.batch_size = args.batch_size


class BasicSuper(BasicModel):
    def __init__(self, args, field_dims, embed_dim):
        super(BasicSuper, self).__init__(args, field_dims, embed_dim)

    def get_random_element_mask(self, batchsize, mask_num):
        input = torch.empty(batchsize, self.embedding_size)
        element_masks = torch.ones_like(input)
        indexes = torch.randint(0, self.embedding_size, size=(batchsize, mask_num))
        element_masks.scatter_(1, indexes, 0)
        element_masks = torch.reshape(element_masks, (element_masks.shape[0], self.field_num, -1)).to(self.device)
        return element_masks

    def calculate_ele_input(self, embed, phase, mask_num):
        xv = embed
        if phase == 'random_train':
            mask_e = self.get_random_element_mask(xv.shape[0], mask_num)
            xe = torch.mul(mask_e, xv)
        if phase == 'test':
            xe = xv
        if phase == 'raw':
            xe = xv
        return xe


class BasicEvo(BasicModel):
    def __init__(self, args, field_dims, embed_dim):
        super(BasicEvo, self).__init__(args, field_dims, embed_dim)

    def calculate_ele_input(self, embed, cand):
        indices = cand.repeat(embed.shape[0], 1)
        embed_x = torch.reshape(embed, (embed.shape[0], -1))
        xe_2d = embed_x.scatter_(1, indices, 0)
        xe = torch.reshape(xe_2d, (xe_2d.shape[0], self.field_num, -1))
        return xe


class BasicRetrain(BasicModel):
    def __init__(self, args, field_dims, embed_dim):
        super(BasicRetrain, self).__init__(args, field_dims, embed_dim)

    def update_ele_mask(self, embed_ele_mask):
        print('embed_ele_mask:')
        print(embed_ele_mask)
        self.embed_ele_indices = embed_ele_mask
        self.embed_ele_indices.requires_grad_(False)

    def update_random_mask(self, mask_num):
        # self.embed_random_index = torch.randint(0, self.embedding_size, size=(1, mask_num))
        self.embed_random_index = torch.randperm(self.embedding_size)[:mask_num]
        self.embed_random_index.requires_grad_(False)
        print('embed_random_mask:')
        print(self.embed_random_index)

    def calculate_ele_input(self, embed):
        indices = self.embed_ele_indices.repeat(embed.shape[0], 1)
        embed_x = torch.reshape(embed, (embed.shape[0], -1))
        xe_2d = embed_x.scatter_(1, indices, 0)
        xe = torch.reshape(xe_2d, (xe_2d.shape[0], self.field_num, -1))
        return xe

    def calculate_rand_input(self, embed):
        indices = self.embed_random_index.repeat(embed.shape[0], 1).to(self.device)
        embed_x = torch.reshape(embed, (embed.shape[0], -1))
        xe_2d = embed_x.scatter_(1, indices, 0)
        xe = torch.reshape(xe_2d, (xe_2d.shape[0], self.field_num, -1))
        return xe
