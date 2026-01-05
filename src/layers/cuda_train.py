# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torch.nn.functional as F


def round_and_to_int8(z):

    z_hat = torch.clamp(torch.round(z), -128., 127.)
    z_hat_write = z_hat.to(dtype=torch.int8)
    return z_hat, z_hat_write


def clamp_reciprocal_with_quant(q_dec, y, min_val):

    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y


def add_and_multiply(y_hat_0, y_hat_1, q_dec):

    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat


def combine_for_reading_2x(x, mask, inplace=False):

    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1


def restore_y_2x(y, means, mask):

    return (torch.cat((y, y), dim=1) + means) * mask


def restore_y_2x_with_cat_after(y, means, mask, to_cat):

    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x(y, means, mask):

    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):

    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):

    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def bias_pixel_shuffle_8(x, bias):

    out = x + bias[None, :, None, None]
    out = F.pixel_shuffle(out, 8)
    out = torch.clamp(out, 0., 1.)
    return out


def bias_quant(x, bias, quant_step):

    out = x + bias[None, :, None, None]
    out = out * quant_step
    return out
