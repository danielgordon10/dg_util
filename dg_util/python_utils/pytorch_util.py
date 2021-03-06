import copy
import glob
import itertools
import numbers
import os
import re
import traceback
from collections import Iterable
from collections import defaultdict, OrderedDict
from itertools import chain
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from PIL import Image
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms

try:
    import accimage
except ImportError:
    accimage = None

DETECT_ANOMALY_COUNT = 0
SURFNORM_KERNEL = None


########## Restore/Save Model Stuff ##########
def restore(net, save_file, saved_variable_prefix="", new_variable_prefix="", skip_filter=None):
    if isinstance(saved_variable_prefix, str):
        saved_variable_prefix = [saved_variable_prefix]
    if isinstance(new_variable_prefix, str):
        new_variable_prefix = [new_variable_prefix]
    print("restoring from", save_file)
    try:
        with torch.no_grad():
            net_state_dict = net.state_dict()
            restore_state_dict = torch.load(save_file, map_location="cpu")
            if type(restore_state_dict) != OrderedDict:
                print("Restored is not OrderedDict, may contain other values.")
                for key, val in restore_state_dict.items():
                    if type(val) == OrderedDict:
                        print("Extracting from", key)
                        restore_state_dict = val
                        break

            restored_var_names = set()
            new_var_names = set()

            print("Restoring:")
            for restore_var_name in restore_state_dict.keys():
                new_var_name = restore_var_name
                changed_name = False
                for svp, nvp in zip(saved_variable_prefix, new_variable_prefix):
                    if len(svp) == 0 and len(nvp) == 0:
                        # Ignore when both are empty, probably just not provided
                        continue
                    if restore_var_name.startswith(svp):
                        original_name = new_var_name
                        new_var_name = nvp + restore_var_name[len(svp) :]
                        changed_name = True
                        break

                if skip_filter is not None and not skip_filter(new_var_name):
                    print("Skipping", new_var_name, "due to skip_filter")
                    continue
                if new_var_name in net_state_dict:
                    var_size = net_state_dict[new_var_name].size()
                    restore_size = restore_state_dict[restore_var_name].size()
                    if var_size != restore_size:
                        print("Shape mismatch for var", restore_var_name, "expected", var_size, "got", restore_size)
                    else:
                        if isinstance(net_state_dict[new_var_name], torch.nn.Parameter):
                            # backwards compatibility for serialized parameters
                            net_state_dict[new_var_name] = restore_state_dict[restore_var_name].data
                        try:
                            net_state_dict[new_var_name].copy_(restore_state_dict[restore_var_name])
                            if changed_name:
                                print(
                                    str(original_name)
                                    + " -> "
                                    + str(new_var_name)
                                    + " -> \t"
                                    + str(var_size)
                                    + " = "
                                    + str(int(np.prod(var_size) * 4 / 10 ** 6))
                                    + "MB"
                                )
                            else:
                                print(
                                    str(restore_var_name)
                                    + " -> \t"
                                    + str(var_size)
                                    + " = "
                                    + str(int(np.prod(var_size) * 4 / 10 ** 6))
                                    + "MB"
                                )
                            restored_var_names.add(restore_var_name)
                            new_var_names.add(new_var_name)
                        except Exception as ex:
                            print(
                                "Exception while copying the parameter named {}, whose dimensions in the model are"
                                " {} and whose dimensions in the checkpoint are {}, ...".format(
                                    restore_var_name, var_size, restore_size
                                )
                            )
                            raise ex

            ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
            unset_var_names = sorted(list(set(net_state_dict.keys()) - new_var_names))
            print("")
            if len(ignored_var_names) == 0:
                print("Restored all variables")
            else:
                print("Did not restore:\n\t" + "\n\t".join(ignored_var_names))
            if len(unset_var_names) == 0:
                print("No new variables")
            else:
                print("Initialized but did not modify:\n\t" + "\n\t".join(unset_var_names))

            print("Restored %s" % save_file)
    except:
        print("Got exception while trying to restore")
        import traceback

        traceback.print_exc()


def get_checkpoint_ind(filename):
    try:
        nums = re.findall(r"\d+", filename)
        start_it = int(nums[-1])
    except:
        traceback.print_exc()
        start_it = 0
        print("Could not parse epoch")
    return start_it


def restore_from_folder(net, folder, saved_variable_prefix="", new_variable_prefix="", skip_filter=None):
    print("restoring from", folder)
    checkpoints = sorted(glob.glob(os.path.join(folder, "*.pt")), key=os.path.getmtime)
    start_it = 0
    if len(checkpoints) > 0:
        restore(net, checkpoints[-1], saved_variable_prefix, new_variable_prefix, skip_filter)
        start_it = get_checkpoint_ind(checkpoints[-1])
    else:
        print("No checkpoints found")
    return start_it


def save(net, file_name, num_to_keep=1, iteration=None):
    if iteration is not None:
        file_name = os.path.join(file_name, "%09d.pt" % iteration)
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    torch.save(net.state_dict(), file_name)
    folder = os.path.dirname(file_name)
    checkpoints = sorted(glob.glob(os.path.join(folder, "*.pt")), key=os.path.getmtime)
    print("Saved %s" % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)


def rename_network_variables(change_dict, filepath, new_basedir):
    state_dict = torch.load(filepath, map_location="cpu")
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        found_match = False
        for bad_key in change_dict.keys():
            if key[: len(bad_key)] == bad_key:
                new_key = change_dict[bad_key] + key[len(bad_key) :]
                print(key + "\t->\t" + new_key)
                new_state_dict[new_key] = val
                found_match = True
                break
        if not found_match:
            new_state_dict[key] = val

    new_path = os.path.join(new_basedir, filepath)
    if not os.path.exists(os.path.dirname(new_path)):
        os.makedirs(os.path.dirname(new_path))
    torch.save(new_state_dict, new_path)


def rename_many_networks_variables(change_dict, basedir, new_basedir="converted"):
    assert basedir != new_basedir, "This may cause problems with os.walk"
    for root, dirs, files in os.walk("logs"):
        for file in files:
            filename = os.path.join(root, file)
            if os.path.splitext(file)[1] == ".pt":
                rename_network_variables(change_dict, filename, new_basedir)


########## Shape stuff ##########
def remove_dim_get_shape(curr_shape, dim):
    assert dim > 0, "Axis must be greater than 0"
    curr_shape = list(curr_shape)
    axis_shape = curr_shape.pop(dim)
    curr_shape[dim - 1] *= axis_shape
    return curr_shape


def remove_dim(input_tensor, dim):
    curr_shape = list(input_tensor.shape)
    if type(dim) == int:
        if dim < 0:
            dim = len(curr_shape) + dim
        new_shape = remove_dim_get_shape(curr_shape, dim)
    else:
        dim = [dd if dd >= 0 else len(curr_shape) + dd for dd in dim]
        assert len(np.unique(dim)) == len(dim), "Repeated dims are not allowed"
        for ax in sorted(dim, reverse=True):
            curr_shape = remove_dim_get_shape(curr_shape, ax)
        new_shape = curr_shape
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor.view(new_shape)
    else:
        return input_tensor.reshape(new_shape)


def split_dim_get_shape(curr_shape, dim, d1, d2):
    assert dim < len(curr_shape), "Axis must be less than the current rank"
    curr_shape.insert(dim, d1)
    curr_shape[dim + 1] = d2
    return curr_shape


def split_dim(input_tensor, dim, d1, d2):
    curr_shape = list(input_tensor.shape)
    if dim < 0:
        dim = len(curr_shape) + dim
    new_shape = split_dim_get_shape(curr_shape, dim, d1, d2)
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor.view(new_shape)
    else:
        return input_tensor.reshape(new_shape)


def expand_dim(tensor: Union[torch.Tensor, np.ndarray], dim: int, desired_dim_len: int) -> torch.Tensor:
    sz = list(tensor.shape)
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))


def expand_new_dim(tensor: Union[torch.Tensor, np.ndarray], new_dim: int, desired_dim_len: int) -> torch.Tensor:
    sz = list(tensor.shape)
    curr_shape = list(tensor.shape)
    curr_shape.insert(new_dim, 1)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.view(curr_shape)
    else:
        tensor = tensor.reshape(curr_shape)
    sz.insert(new_dim, desired_dim_len)
    return tensor.expand(tuple(sz))


def repeat_new_dim(tensor: torch.Tensor, new_dim: int, desired_dim_len: int) -> torch.Tensor:
    sz = [1 for _ in range(len(tensor.size()))]
    curr_shape = list(tensor.size())
    curr_shape.insert(new_dim, 1)
    tensor = tensor.view(curr_shape)
    sz.insert(new_dim, desired_dim_len)
    return tensor.repeat(tuple(sz))


########## From/To Numpy ##########
def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, dict):
        return {key: to_numpy(val) for key, val in array.items()}
    elif isinstance(array, list) and isinstance(array[0], torch.Tensor):
        array = [to_numpy(val) for val in array]
        return np.asarray(array)
    else:
        return np.asarray(array)


numpy_dtype_to_pytorch_dtype_warn = False


def numpy_dtype_to_pytorch_dtype(numpy_dtype):
    global numpy_dtype_to_pytorch_dtype_warn
    # Extremely gross conversion but the only one I've found
    numpy_dtype = np.dtype(numpy_dtype)
    if numpy_dtype == np.uint32:
        if not numpy_dtype_to_pytorch_dtype_warn:
            print("numpy -> torch dtype uint32 not supported, using int32")
            numpy_dtype_to_pytorch_dtype_warn = True
        numpy_dtype = np.int32
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).detach().dtype


from_numpy_warn = defaultdict(lambda: False)


def from_numpy(np_array):
    global from_numpy_warn
    if isinstance(np_array, list) or isinstance(np_array, tuple):
        try:
            np_array = np.stack(np_array, 0)
        except ValueError:
            np_array = np.stack([from_numpy(val) for val in np_array], 0)
    elif isinstance(np_array, dict):
        return {key: from_numpy(val) for key, val in np_array.items()}
    np_array = np.asarray(np_array)
    if np_array.dtype == np.uint32:
        if not from_numpy_warn[np.uint32]:
            print("numpy -> torch dtype uint32 not supported, using int32")
            from_numpy_warn[np.uint32] = True
        np_array = np_array.astype(np.int32)
    elif np_array.dtype == np.dtype("O"):
        if not from_numpy_warn[np.dtype("O")]:
            print("numpy -> torch dtype Object not supported, returning numpy array")
            from_numpy_warn[np.dtype("O")] = True
        return np_array
    elif np_array.dtype.type == np.str_:
        if not from_numpy_warn[np.str_]:
            print("numpy -> torch dtype numpy.str_ not supported, returning numpy array")
            from_numpy_warn[np.str_] = True
        return np_array
    return torch.from_numpy(np_array)


def stack_dicts_in_list(dicts, axis=0, concat=False):
    keys = set(itertools.chain.from_iterable(dict.keys() for dict in dicts))
    stacked_dict = {}
    for key in keys:
        vals = [dict[key] for dict in dicts if key in dict]
        if len(vals) == 1:
            if concat:
                vals = vals[0]
            else:
                vals = vals[0]
                if isinstance(vals, np.ndarray):
                    vals = np.expand_dims(vals, axis)
                elif isinstance(vals, torch.Tensor):
                    vals.unsqueeze_(axis)
        else:
            if isinstance(vals[0], np.ndarray):
                if concat:
                    vals = np.concatenate(vals, axis=axis)
                else:
                    vals = np.stack(vals, axis=axis)
            elif isinstance(vals[0], torch.Tensor):
                if concat:
                    vals = torch.cat(vals, dim=axis)
                else:
                    vals = torch.stack(vals, dim=axis)
        stacked_dict[key] = vals
    return stacked_dict


############### Layers ###############
class Identity(nn.Module):
    def forward(self, data):
        return data


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super(LambdaLayer, self).__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


class RemoveDim(nn.Module):
    def __init__(self, dim):
        super(RemoveDim, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        return remove_dim(input_tensor, self.dim)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1, nonlinearity=nn.ReLU):
    layers = [
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    ]
    if nonlinearity is not None:
        layers.append(nonlinearity())
    return nn.Sequential(*layers)


def conv_block(in_planes, out_planes, kernel_size=3, padding=None, stride=1, norm_layer=nn.BatchNorm2d, nonlinearity=nn.ReLU):
    if padding is None:
        padding = kernel_size // 2
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding)]
    if norm_layer is not None:
        layers.append(norm_layer(out_planes))
    if nonlinearity is not None:
        layers.append(nonlinearity())
    return nn.Sequential(*layers)


def linear_block(in_features, out_features, norm_layer=nn.BatchNorm2d, nonlinearity=nn.ReLU, dropout=0):
    layers = [nn.Linear(in_features, out_features)]
    if norm_layer is not None:
        layers.append(norm_layer(out_features))
    if nonlinearity is not None:
        layers.append(nonlinearity())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def upsample_add(x, y):
    if x.shape[2:] != y.shape[2:]:
        _, _, height, width = y.size()
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False) + y
    else:
        return x + y


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )


class AttentionPool2D(nn.Module):
    def __init__(self, feature_size, keepdim=True, return_masks=True):
        super(AttentionPool2D, self).__init__()
        self.attention = nn.Conv2d(feature_size, 1, 1)
        self.keepdim = keepdim
        self.return_masks = return_masks

    def forward(self, x: torch.Tensor):
        x_weight = self.attention(x)
        x_weight_shape = x_weight.shape
        x_weight = remove_dim(x_weight, (2, 3))
        x_weight = F.softmax(x_weight, dim=-1)
        x_weight = x_weight.view(x_weight_shape)
        x = x * x_weight
        x = x.sum((2, 3), keepdim=self.keepdim)
        if self.return_masks:
            return x, x_weight
        else:
            return x


############# Network Stuff #############
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self._device = None
        self.saves = 0

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        self._device = None
        model = super(BaseModel, self).to(*args, **kwargs)
        device = self.device
        return model

    @property
    def name(self):
        return type(self).__name__

    def restore(self, checkpoint_dir, saved_variable_prefix=None, new_variable_prefix=None, skip_filter=None) -> int:
        iteration = restore_from_folder(
            self, os.path.join(checkpoint_dir, "*"), saved_variable_prefix, new_variable_prefix, skip_filter
        )
        return iteration


def reset_module(module):
    module_list = [sub_mod for sub_mod in module.modules()]
    ss = 0
    while ss < len(module_list):
        sub_mod = module_list[ss]
        if hasattr(sub_mod, "reset_parameters"):
            sub_mod.reset_parameters()
            ss += len([_ for _ in sub_mod.modules()])
        else:
            ss += 1


def replace_all_layer_with_new_type(model, existing_layer_type, out_layer_fn):
    modules = model._modules
    for m in modules.keys():
        module = modules[m]
        should_replace = False
        if isinstance(existing_layer_type, Iterable):
            for etype in existing_layer_type:
                if isinstance(module, etype):
                    should_replace = True
                    break
        elif isinstance(module, existing_layer_type):
            should_replace = True
        if should_replace:
            out_layer = out_layer_fn(module)
            print("Old ", m, module, "->", out_layer)
            model._modules[m] = out_layer
        elif isinstance(module, nn.Module):
            model._modules[m] = replace_all_layer_with_new_type(module, existing_layer_type, out_layer_fn)
    return model


def get_first_conv_layer(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.Conv2d):
        return module
    conv_layer = None

    for sub_module in module.children():
        if isinstance(sub_module, nn.Conv2d):
            conv_layer = sub_module
            break
        else:
            deeper = get_first_conv_layer(sub_module)
            if deeper is not None:
                conv_layer = deeper
                break
    return conv_layer


def get_last_conv_layer(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.Conv2d):
        return module
    conv_layer = None
    for sub_module in module.children():
        if isinstance(sub_module, nn.Conv2d):
            conv_layer = sub_module
        else:
            deeper = get_last_conv_layer(sub_module)
            if deeper is not None:
                conv_layer = deeper
    return conv_layer


def get_stride(module: nn.Module) -> Optional[nn.Module]:
    stride = 1
    if hasattr(module, "stride"):
        if isinstance(module.stride, int):
            stride *= module.stride
        else:
            stride *= module.stride[0]
    for sub_module in module.children():
        if hasattr(sub_module, "stride"):
            if isinstance(sub_module.stride, int):
                stride *= sub_module.stride
            else:
                stride *= sub_module.stride[0]
        else:
            deeper = get_stride(sub_module)
            stride *= deeper
    return stride


########## Data/Preprocessing Utils ##########
def fix_broadcast(input1, input2):
    original_shape1 = input1.shape
    original_shape2 = input2.shape
    swap_order = False
    # shape1 should be the longer one
    if len(input1.shape) < len(input2.shape):
        swap_order = True
        input1, input2 = input2, input1
    shape1 = list(input1.shape[::-1])
    shape2 = list(input2.shape[::-1])
    for ii in range(len(shape1)):
        if ii >= len(shape2) or shape1[ii] != shape2[ii]:
            shape2.insert(ii, 1)

    assert len(shape1) == len(shape2), (
        "shapes " + str(original_shape1) + " and ",
        str(original_shape2) + " are not broadcast-fixable.",
    )
    shape2 = shape2[::-1]
    if isinstance(input2, np.ndarray):
        input2 = input2.reshape(shape2)
    else:
        input2 = input2.view(shape2)
    if swap_order:
        input1, input2 = input2, input1
    return input1, input2


def normalize(input_tensor, mean, std):
    mean = from_numpy(mean).to(input_tensor.device, input_tensor.dtype)
    std = from_numpy(std).to(input_tensor.device, input_tensor.dtype)
    input_tensor, mean = fix_broadcast(input_tensor, mean)
    input_tensor, std = fix_broadcast(input_tensor, std)
    input_tensor = input_tensor - mean
    input_tensor = input_tensor / std
    return input_tensor


def unnormalize(input_tensor, mean, std):
    mean = from_numpy(mean).to(input_tensor.device, input_tensor.dtype)
    std = from_numpy(std).to(input_tensor.device, input_tensor.dtype)
    input_tensor, mean = fix_broadcast(input_tensor, mean)
    input_tensor, std = fix_broadcast(input_tensor, std)
    input_tensor = input_tensor * std
    input_tensor = input_tensor + mean
    return input_tensor


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToPILImage(transforms.ToPILImage):
    def __call__(self, pic):
        if _is_pil_image(pic):
            return pic
        return torchvision.transforms.functional.to_pil_image(pic, self.mode)


class ToTensor(object):
    def __init__(self, transpose=True, scale=255):
        self.transpose = transpose
        self.scale = scale

    def __call__(self, pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

        See ``ToTensor`` for more details.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            if self.transpose:
                pic = pic.transpose((2, 0, 1))
            img = torch.from_numpy(pic)
            if self.scale is not None:
                return img.float().div(self.scale)
            return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == "F":
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == "1":
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        if self.transpose:
            img = img.permute(2, 0, 1).contiguous()
        if self.scale is not None:
            return img.float().div(self.scale)
        else:
            return img


class IndexWrapperDataset(Dataset):
    def __init__(self, other_dataset: Dataset):
        self.other_dataset = other_dataset

    def __str__(self):
        return "IndexWrapperDataset: " + str(self.other_dataset)

    def __len__(self):
        return len(self.other_dataset)

    def __repr__(self):
        return "IndexWrapperDataset: " + repr(self.other_dataset)

    def __getitem__(self, item):
        result = self.other_dataset[item]
        return result, item


########## Loss Stuff ##########
def weighted_loss(loss_function_output, weights, reduction="mean"):
    if isinstance(weights, numbers.Number):
        if reduction == "mean":
            return weights * torch.mean(loss_function_output)
        elif reduction == "sum":
            return weights * torch.sum(loss_function_output)
        else:
            return weights * loss_function_output

    elif weights.dtype == torch.uint8 and reduction != "none":
        if reduction == "mean":
            return torch.mean(torch.masked_select(loss_function_output, weights))
        else:
            return torch.sum(torch.masked_select(loss_function_output, weights))
    else:
        if reduction == "mean":
            return torch.mean(loss_function_output * weights)
        elif reduction == "sum":
            return torch.sum(loss_function_output * weights)
        else:
            return loss_function_output * weights


def multi_class_cross_entropy_loss(predictions, labels, reduction="mean", dim=-1):
    # Predictions should be logits, labels should be probabilities.
    loss = labels * F.log_softmax(predictions, dim=dim)
    if reduction == "none":
        return -1 * loss
    elif reduction == "mean":
        return -1 * torch.mean(loss) * predictions.shape[dim]  # mean across all dimensions except softmax one.
    elif reduction == "sum":
        return -1 * torch.sum(loss)
    else:
        raise NotImplementedError("Not known reduction type")


def triplet_ratio_loss(dist_pos: torch.Tensor, dist_neg: torch.Tensor, margin: float = 1e-5) -> torch.Tensor:
    ratio = dist_neg / torch.clamp_min(dist_pos + margin, 1e-10)
    ratio = torch.clamp_min(1 - ratio, 0)
    return ratio


def triplet_ratio_loss_exp(dist_pos: torch.Tensor, dist_neg: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
    # https://www.researchgate.net/profile/Krystian_Mikolajczyk/publication/317192886_Learning_local_feature_descriptors_with_triplets_and_shallow_convolutional_neural_networks/links/5a038dad0f7e9beb1770c3c2/Learning-local-feature-descriptors-with-triplets-and-shallow-convolutional-neural-networks.pdf
    dist_pos = dist_pos / temperature
    dist_neg = dist_neg / temperature
    pos_exp = torch.exp(dist_pos)
    neg_exp = torch.exp(dist_neg)
    denom = pos_exp + neg_exp
    loss = torch.pow((pos_exp / denom), 2) + torch.pow((1 - neg_exp / denom), 2)
    return loss


########## Other ##########
class DataParallelFix(nn.DataParallel):
    """
    Temporary workaround for https://github.com/pytorch/pytorch/issues/15716.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._replicas = None
        self._outputs = None

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    "module must have its parameters and buffers "
                    "on device {} (device_ids[0]) but found one of "
                    "them on device: {}".format(self.src_device_obj, t.device)
                )

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        self._replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        self._outputs = self.parallel_apply(self._replicas, inputs, kwargs)

        return self.gather(self._outputs, self.output_device)


class DummyScope(nn.Module):
    """Used for keeping scope the same between pretrain and interactive training."""

    def __init__(self, module, scope_list):
        super(DummyScope, self).__init__()
        assert isinstance(scope_list, list) and len(scope_list) > 0
        self.scope_list = scope_list
        if len(scope_list) > 1:
            setattr(self, scope_list[0], DummyScope(module, scope_list[1:]))
        elif len(scope_list) == 1:
            setattr(self, scope_list[0], module)

    def forward(self, *input, **kwargs):
        return getattr(self, self.scope_list[0])(*input, **kwargs)

def get_data_parallel(module, device_ids):
    if isinstance(device_ids, str):
        device_ids = [int(device_id.strip()) for device_id in device_ids.split(",")]
    if device_ids is None or len(device_ids) == 1:
        return DummyScope(module, ["module"])
    else:
        print("Torch using", len(device_ids), "GPUs", device_ids)
        return nn.DataParallel(module, device_ids=device_ids)


def detatch_recursive(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detatch_recursive(v) for v in h)


def get_one_hot(data, num_inds, dtype=torch.float32):
    assert (data.max() < num_inds).item()
    placeholder = torch.zeros(data.shape + (num_inds,), device=data.device, dtype=dtype)
    placeholder_shape = placeholder.shape
    placeholder = placeholder.view(-1, num_inds)
    placeholder[torch.arange(data.numel()), data.view(-1)] = 1
    placeholder = placeholder.view(placeholder_shape)
    return placeholder


def get_one_hot_numpy(data, num_inds, dtype=np.float32):
    data = np.asarray(data)
    assert data.max() < num_inds
    placeholder = np.zeros(data.shape + (num_inds,), dtype=dtype)
    placeholder_shape = placeholder.shape
    placeholder = placeholder.reshape(-1, num_inds)
    placeholder[np.arange(data.size), data.reshape(-1)] = 1
    placeholder = placeholder.reshape(placeholder_shape)
    return placeholder


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    global SURFNORM_KERNEL
    if SURFNORM_KERNEL is None:
        SURFNORM_KERNEL = torch.from_numpy(
            np.array(
                [
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            )
        )[:, np.newaxis, ...].to(dtype=torch.float32, device=depth.device)
    with torch.no_grad():
        surface_normals = F.conv2d(depth, surfnorm_scalar * SURFNORM_KERNEL, padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(dim=1, keepdim=True)
    return surface_normals












def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class detect_anomaly(object):
    def __init__(self, frequency):
        self.prev = torch.is_anomaly_enabled()
        self.frequency = frequency

    def __enter__(self):
        global DETECT_ANOMALY_COUNT
        if DETECT_ANOMALY_COUNT % self.frequency == 0:
            torch.set_anomaly_enabled(True)
        DETECT_ANOMALY_COUNT += 1

    def __exit__(self, *args):
        torch.set_anomaly_enabled(self.prev)
        return False





