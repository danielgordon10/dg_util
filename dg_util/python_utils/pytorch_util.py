import glob
import numbers
import os
import re
import traceback
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


def restore(net, save_file, saved_variable_prefix="", new_variable_prefix="", skip_filter=None):
    if isinstance(saved_variable_prefix, str):
        saved_variable_prefix = [saved_variable_prefix]
    if isinstance(new_variable_prefix, str):
        new_variable_prefix = [new_variable_prefix]
    print("restoring from", save_file)
    try:
        with torch.no_grad():
            net_state_dict = net.state_dict()
            if torch.cuda.is_available():
                device = next(net.parameters()).device
                restore_state_dict = torch.load(save_file, device)
            else:
                restore_state_dict = torch.load(save_file, map_location="cpu")

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
                        new_var_name = nvp + restore_var_name[len(svp):]
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


def remove_dim_get_shape(curr_shape, dim):
    assert dim > 0, "Axis must be greater than 0"
    curr_shape = list(curr_shape)
    axis_shape = curr_shape.pop(dim)
    curr_shape[dim - 1] *= axis_shape
    return curr_shape


def remove_dim(input_tensor, dim):
    curr_shape = list(input_tensor.shape)
    if type(dim) == int:
        new_shape = remove_dim_get_shape(curr_shape, dim)
    else:
        for ax in sorted(dim, reverse=True):
            curr_shape = remove_dim_get_shape(curr_shape, ax)
        new_shape = curr_shape
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor.view(new_shape)
    else:
        return input_tensor.reshape(new_shape)


class RemoveDim(nn.Module):
    def __init__(self, dim):
        super(RemoveDim, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        return remove_dim(input_tensor, self.dim)


def split_axis_get_shape(curr_shape, axis, d1, d2):
    assert axis < len(curr_shape), "Axis must be less than the current rank"
    curr_shape.insert(axis, d1)
    curr_shape[axis + 1] = d2
    return curr_shape


def split_axis(input_tensor, axis, d1, d2):
    curr_shape = list(input_tensor.shape)
    new_shape = split_axis_get_shape(curr_shape, axis, d1, d2)
    return input_tensor.view(new_shape)


def detatch_recursive(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detatch_recursive(v) for v in h)


def to_numpy_array(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    elif isinstance(array, dict):
        return {key: to_numpy_array(val) for key, val in array.items()}
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


surfnorm_kernel = None


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    global surfnorm_kernel
    if surfnorm_kernel is None:
        surfnorm_kernel = torch.from_numpy(
            np.array(
                [
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            )
        )[:, np.newaxis, ...].to(dtype=torch.float32, device=depth.device)
    with torch.no_grad():
        surface_normals = F.conv2d(depth, surfnorm_scalar * surfnorm_kernel, padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(dim=1, keepdim=True)
    return surface_normals


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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
        return torch.nn.DataParallel(module, device_ids=device_ids)


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


class ToTensor(object):
    def __init__(self, transpose=True, scale=None):
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
