import torch
import torch.nn as nn
from torch import einsum

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import rasterio
from sklearn.model_selection import train_test_split

# import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import albumentations

import pandas as pd
import numpy as np
import math

device = torch.device("cuda")


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    ##print(str(x1.shape) + ' x2 = ' + str(x2.shape))
    ##print(f'this function will return a shape of shape {torch.cat((-x2, x1), dim = -1).shape}')
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(freqs, t, sindex=0):
    rot_dim = freqs.shape[-1]
    # print(f'we (a_r_e) received a freq shape of {freqs.shape} and rotation dimension is of size {rot_dim}')
    eindex = sindex + rot_dim
    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (
        t[..., :sindex],
        t[..., sindex:eindex],
        t[..., eindex:],
    )
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)


def mixup_data(Images, y, tab, alpha=0.1, p=0.85, use_cuda=True):

    """
    Compute the 'partially' mixed up data.
    P is probability of mixup being applied

    Return: mixed inputs, pairs of targets, and lambda.
    """

    batch_size = Images.size()[0]
    mix_items = (np.random.binomial(n=1, size=batch_size, p=p) + 1) % 2

    tab = tab.cuda()
    Images = Images.cuda()
    y = y.cuda()

    if alpha > 0.0:
        lam = 1 - np.random.beta(alpha, alpha, size=Images.size()[0]) * mix_items
        lam = torch.from_numpy(lam).cuda()
    else:
        lam = 1.0

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = torch.einsum("b, bxyc -> bxyc", lam, Images) + torch.einsum(
        "b, bxyc -> bxyc", (1 - lam), Images[index]
    )
    mixed_y = lam * y + (1 - lam) * y[index]
    mixed_tab = torch.einsum("b, bd -> bd", lam, tab) + torch.einsum(
        "b, bd -> bd", (1 - lam), tab[index]
    )
    return mixed_x.float(), mixed_y.float(), mixed_tab.float()


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        # return self.fn(nn.utils.weight_norm(x), **kwargs)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10, num_freqs=1, device=torch.device("cuda")):
        super().__init__()
        self.freqs = (
            torch.logspace(
                0.0,
                math.log(max_freq / 2) / math.log(2),
                dim // 2,
                base=2,
                device=device,
            )
            * math.pi
        )
        self.cache = dict()
        self.device = device

    def forward(self, t, cache_key=None):
        freqs = self.freqs
        t.to(self.device)
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype).cuda(), freqs)

        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if cache_key is not None:
            self.cache[cache_key] = freqs

        return freqs


def tifffile_loader(path):

    """
    Helper function to read tiffs into python and do some error handling

    all the loader should be numpy ndarray [height (1001), width (1001), channels (32)]
     Note: int16: (-32768 to 32767)
    """

    import tifffile

    img = tifffile.imread(path)
    if img.dtype in [np.uint8, np.uint16, np.float]:
        return img
    else:
        raise TypeError(
            "tiff file only support np.uint8, np.uint16, np.float, but got {}".format(
                img.dtype
            )
        )


class FireManagement_Dataset(Dataset):
    """
    Torch Dataset to hold satellite raster data, tabular data and time-series raster data

    Needs - directory for pickled tabular data (pickle_file)
          - raster_dir (directory with fire-relevant rasters)
          - transform (augmentations for images)
          - outcome (tabular dataframe label we are trying to predict)

    Assumptions - you use IrwinID to index your fire data. This means it won't work if you
    try to use this dataset on non-irwin tabular fire data. To fix this, pass your unique identifier to
    the 'uid' parameter. This requires you followed the same naming convention for your fire files as well.

    Will return your GACC, IrwinID, your target, number of resources assigned to other fires at time of ignition,
    numeric month, number of days from ignition to response, and an indicator for fires that begin as 'full suppression'
    """

    def __init__(
        self,
        pickle_file,
        raster_dir,
        outcome,
        transform,
        image_loader=tifffile_loader,
        log_label=False,
        mixup=True,
        mixup_p=0.05,
        train=True,
        split_frac=None,
        fold=None,
        uid="IrwinID",
    ):
        self.raster_dir = raster_dir
        self.pickle_file = pickle_file
        self.tab = pd.read_pickle(pickle_file)
        if split_frac == None:
            split = pd.read_csv("fold_labels.csv")
            train_data = self.tab[
                self.tab[uid].isin(split.loc[split["fold"] != fold][uid])
            ]
            valid_data = self.tab[self.tab[uid].isin(split[split["fold"] == fold][uid])]
        else:
            train_data, valid_data = train_test_split(
                self.tab, test_size=split_frac, random_state=6652
            )
        if train:
            self.tabular = train_data
        else:
            self.tabular = valid_data
        self.outcome = outcome
        self.transform = transform
        self.image_loader = image_loader
        self.log_label = log_label
        self.mixup = mixup
        self.train = train
        self.uid = uid
        if mixup:
            self.mixup_p = mixup_p
        super().__init__()

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            label, image = self.tabular.iloc[idx, 0:][self.outcome], self.image_loader(
                f"{self.raster_dir}/{self.tabular.iloc[idx, 0:][self.uid]}/{self.tabular.iloc[idx, 0:][self.uid]}.tif"
            )
            # print(label)
            # print(self.tabular.iloc[idx, 0:]['IrwinID'])
        tabular = self.tabular.iloc[idx].loc[
            self.tabular.columns.isin(
                [
                    "DaysToDiscovery",
                    "GACC_EACC",
                    "GACC_GBCC",
                    "GACC_NRCC",
                    "GACC_NWCC",
                    "GACC_ONCC",
                    "GACC_OSCC",
                    "GACC_RMCC",
                    "GACC_SACC",
                    "GACC_SWCC",
                    "GACC_nan",
                    self.outcome,
                    self.uid,
                    "InitialFireStrategy_Full Suppression",
                    "day_of_year",
                    "norm_res",
                ]
            )
        ]

        y = tabular[self.outcome]

        if self.log_label:
            y = np.log(y + 1)

        # with rasterio.open(f"{self.raster_dir}/{tabular['IrwinID']}/{tabular['IrwinID']}.tif") as set:
        #    image = set.read()

        path = f"{self.raster_dir}/{tabular[self.uid]}/{tabular[self.uid]}.tif"
        image = self.image_loader(path)

        image[image < -9998] = 0

        image[image > 100000000] = 0

        image[np.isnan(image)] = 0

        maxes = np.load(
            "/home/connor/Desktop/MassCrop/OutputFolder/raster/rastmax.npy",
            allow_pickle=True,
        )
        maxes[maxes == 0] = 1
        mins = np.load(
            "/home/connor/Desktop/MassCrop/OutputFolder/raster/rastmin.npy",
            allow_pickle=True,
        )

        image = (image - mins) / (maxes - mins)

        image = np.float32(image)

        # print(image)

        # print(image.shape)
        # image = np.transpose(image,[1,0,2])
        # print(image.shape)

        nx, ny = image[:, :, 0].shape
        x_dim = (
            np.arange(nx) - (nx - 1) / 2.0
        )  # x an y so they are distance from center, assuming array is "nx" long (as opposed to 1. which is the other common choice)
        y_dim = np.arange(ny) - (ny - 1) / 2.0
        X, Y = np.meshgrid(x_dim, y_dim)
        d = np.sqrt(X ** 2 + Y ** 2) / np.sqrt(nx ** 2 + ny ** 2)
        image = np.dstack([image, d / 0.5])

        if self.transform is not None and self.train:
            image = self.transform(image=image)["image"]

        image = torch.from_numpy(image)

        ch = image.shape.index(34)
        image = image.permute(ch, ch - 1, ch - 2)
        # print(image.shape)
        image = image.float()

        # shape = (28 (27 plus dist from center),1001,1001)

        # image = torch.cat((image, torch.from_numpy(d).unsqueeze(0)))
        IrwinID = tabular[self.uid]
        tabular = tabular.drop(labels=[self.uid, self.outcome]).fillna(0)

        return (
            np.array(image.squeeze()),
            torch.tensor(tabular.values.astype(np.float32)).cuda(),
            y.astype(np.float32),
            IrwinID,
        )


class SingleItemRetriever:
    """Utility function to return single (specific) image"""

    def __init__(
        self,
        pickle_file,
        raster_dir,
        outcome,
        image_loader=tifffile_loader,
        log_label=False,
        label=None,
    ):
        self.raster_dir = raster_dir
        self.pickle_file = pickle_file
        self.tab = pd.read_pickle(pickle_file)

        self.tabular = self.tab[self.tab["IrwinID"] == label]

        self.outcome = outcome
        self.idnam = label

        self.image_loader = image_loader
        self.log_label = log_label

    def ret(self):
        label, image = self.tabular[self.outcome], self.image_loader(
            f"{self.raster_dir}/{self.idnam}/{self.idnam}.tif"
        )
        # print(label)
        # print(self.tabular.iloc[idx, 0:]['IrwinID'])
        tabular = self.tabular.loc[
            :,
            self.tabular.columns.isin(
                [
                    "DaysToDiscovery",
                    "GACC_EACC",
                    "GACC_GBCC",
                    "GACC_NRCC",
                    "GACC_NWCC",
                    "GACC_ONCC",
                    "GACC_OSCC",
                    "GACC_RMCC",
                    "GACC_SACC",
                    "GACC_SWCC",
                    "GACC_nan",
                    self.outcome,
                    "IrwinID",
                    "InitialFireStrategy_Full Suppression",
                    "day_of_year",
                    "norm_res",
                ]
            ),
        ]
        y = tabular[self.outcome]

        if self.log_label:
            y = np.log(y + 1)

        # with rasterio.open(f"{self.raster_dir}/{tabular['IrwinID']}/{tabular['IrwinID']}.tif") as set:
        #    image = set.read()

        path = f"{self.raster_dir}/{self.idnam}/{self.idnam}.tif"
        image = self.image_loader(path)

        image[image < -9998] = 0

        image[image > 100000000] = 0

        image[np.isnan(image)] = 0

        maxes = np.load(
            "/home/connor/Desktop/MassCrop/OutputFolder/raster/rastmax.npy",
            allow_pickle=True,
        )
        maxes[maxes == 0] = 1
        mins = np.load(
            "/home/connor/Desktop/MassCrop/OutputFolder/raster/rastmin.npy",
            allow_pickle=True,
        )

        image = (image - mins) / (maxes - mins)

        image = np.float32(image)

        # print(image)

        # print(image.shape)
        # image = np.transpose(image,[1,0,2])
        # print(image.shape)

        nx, ny = image[:, :, 0].shape
        x_dim = (
            np.arange(nx) - (nx - 1) / 2.0
        )  # x an y so they are distance from center, assuming array is "nx" long (as opposed to 1. which is the other common choice)
        y_dim = np.arange(ny) - (ny - 1) / 2.0
        X, Y = np.meshgrid(x_dim, y_dim)
        d = np.sqrt(X ** 2 + Y ** 2) / np.sqrt(nx ** 2 + ny ** 2)
        image = np.dstack([image, d / 0.5])

        image = torch.from_numpy(image)

        ch = image.shape.index(34)
        image = image.permute(ch, ch - 1, ch - 2)
        # print(image.shape)
        image = image.float()

        # shape = (28 (27 plus dist from center),1001,1001)

        # image = torch.cat((image, torch.from_numpy(d).unsqueeze(0)))

        tabular = tabular.loc[
            :, ~tabular.columns.isin(["IrwinID", self.outcome])
        ].fillna(0)

        return (
            torch.from_numpy(np.array(image.squeeze())),
            torch.tensor(tabular.values.astype(np.float32)).cuda(),
            y.astype(np.float32),
        )


class Patching(nn.Module):
    """
    Utility fcn that patches the full image
    """

    def __init__(
        self,
        in_channels=22,
        patch_size=143,
        embedding_dim=768,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embedding_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
            ),
            nn.Flatten(2, 3),
        )

    def forward(self, x):
        x = rearrange(self.patch(x).transpose(-2, -1), "b c x y -> b c (x y)")
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            MultiheadedSelfAttention(
                                dim, num_heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiheadedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        pos_embedding=RotaryEmbedding(dim=200),
        num_heads=21,
        dim_head=600,
        dropout=0.0,
        channels=27,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1).cuda()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False).cuda()

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.pos_embedding = pos_embedding

    def forward(self, x):

        b, n, _, h = *x.shape, self.num_heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        cls_tokens = q[:, :, 0, :]

        q, k, v = map(lambda t: t[:, :, 1:, :], (q, k, v))

        # print('queries!')
        # print(f'queries look like - {q.shape}')
        # print('keys!')
        # print(f'keys look like - {k.shape}')

        # print((einsum('b h i d, b h j d -> b h i j', q, k) * self.scale).shape)

        freqs_h = self.pos_embedding(
            torch.linspace(-1, 1, steps=50), cache_key=50
        )  # 13
        freqs_w = self.pos_embedding(torch.linspace(-1, 1, steps=50), cache_key=50)
        freqs = self.broadcat(
            (freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1
        )

        # print('freqs!')
        # print(f'freqs.shape is - {freqs.shape}')

        q = apply_rotary_emb(
            freqs, rearrange(q, "b h (n z) d -> (b h) n z d", n=50)
        )  # 13
        k = apply_rotary_emb(freqs, rearrange(k, "b h (n z) d -> (b h) n z d", n=50))

        q = rearrange(q, "(b h) n z d -> b h (n z) d", h=self.heads)
        k = rearrange(k, "(b h) n z d -> b h (n z) d", h=self.heads)

        cls_tokens = cls_tokens[:, :, None, :]
        q, k, v = map(lambda t: torch.cat((cls_tokens, t), dim=2), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def broadcat(self, tensors, dim=-1):
        # print(f'tensors begin life looking like... {tensors[0].shape} and {tensors[1].shape}')
        num_tensors = len(tensors)
        shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
        assert (
            len(shape_lens) == 1
        ), "tensors must all have the same number of dimensions"
        shape_len = list(shape_lens)[0]

        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*map(lambda t: list(t.shape), tensors)))
        # print(f'dims is a list that looks like {dims}')

        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all(
            [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
        ), "invalid dimensions for broadcastable concatentation"
        max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
        expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
        tensors = list(
            map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes))
        )
        return torch.cat(tensors, dim=dim)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ).cuda()

    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=(500, 500),  # (1001, 1001) before patching
        patch_size=(10, 10),
        num_classes=3,
        dim=800,
        fe_dim=14,
        depth=3,
        num_heads=10,
        mlp_dim=1000,
        pool="mean",
        channels=110,
        dim_head=600,
        dropout=0.1,
        emb_dropout=0.0,
        batch_size=1,
        learned_pool=None,
    ):
        super(ViT, self).__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be multiple of patch size."

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pooling must be done in either a classification token manner (pool = cls) or by averaging the heads (pool = mean) (mean pooling)"

        self.to_patch_embedding_step_1 = Tokenizer()
        self.to_patch_embedding_step_2 = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout).cuda()

        self.transformer = TransformerEncoderLayer(
            dim, depth, num_heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity().cuda()

        self.to_latent_2 = nn.Linear(dim_head * num_heads, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 5),
        )
        self.mlp_head_1 = nn.Sequential(
            nn.Linear(dim // 5 + fe_dim, dim // 5),
        )

        self.mlp_head_out = nn.Linear(dim // 5, num_classes)
        self.SELU = nn.SELU()

        self.ln = nn.LayerNorm(dim // 10)

        self.batch_size = batch_size

    def forward(self, img, tabular):

        x = self.to_patch_embedding_step_1(img)
        x = self.to_patch_embedding_step_2(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        # concat data for post-transformer learning
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.mlp_head(x)

        x = self.SELU(x)

        x = self.mlp_head_1(torch.cat((tabular, x), dim=1))
        x = self.SELU(x)
        x = self.mlp_head_out(x)
        return x


class Tokenizer(nn.Module):
    def __init__(
        self,
        kernel_size=13,
        kernel_size_2=3,
        stride=1,
        stride_2=2,
        padding_2="valid",
        padding="same",
        pooling_kernel_size=2,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=34,
        n_output_channels=110,
        in_planes=68,
    ):
        super(Tokenizer, self).__init__()

        n_filter_list = (
            [n_input_channels]
            + [in_planes for _ in range(n_conv_layers - 1)]
            + [n_output_channels]
        )

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.PReLU(n_output_channels),
                    torch.nn.AdaptiveMaxPool2d((500, 500)),
                )
                for i in range(n_conv_layers)
            ]
        )

    def forward(self, x):
        return self.conv_layers(x).transpose(-2, -1)


def TrainerTransform(img):

    train_transform = albumentations.Compose(
        [
            albumentations.CoarseDropout(max_holes=17, max_height=8),
            albumentations.ShiftScaleRotate(
                shift_limit=0.00, scale_limit=0.00, rotate_limit=45, p=0.1
            ),
            albumentations.GaussianBlur(
                blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5
            ),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.RandomRotate90(),
        ]
    )

    return train_transform(image=img)["image"]


def QuantileLoss(input, target, q1=0.1, q2=0.5, q3=0.9):

    ## Quanile Loss
    q1 = q1
    q2 = q2
    q3 = q3

    e1 = input[:, 0:1] - target
    e2 = input[:, 1:2] - target
    e3 = input[:, 2:3] - target
    eq1 = torch.max(q1 * e1, (q1 - 1) * e1)
    eq2 = torch.max(q2 * e2, (q2 - 1) * e2)
    eq3 = torch.max(q3 * e3, (q3 - 1) * e3)

    loss = (eq1 + eq2 + eq3).mean()

    return loss
