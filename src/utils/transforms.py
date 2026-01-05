import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb_to_ycbcr420(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/2), in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    cb = np.mean(np.reshape(cb, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    cr = np.mean(np.reshape(cr, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def rgb_to_ycbcr444(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2xhxw, in the range of [0, 1]
    '''
    c, _, _ = rgb.shape
    assert c == 3
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def ycbcr420_to_rgb(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def ycbcr444_to_rgb(y, uv):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2xhxw UV float numpy array, in the range of [0, 1]
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def ycbcr420_to_444(y, uv, order=0, separate=False):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    '''
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    '''
    y is 1xhxw Y float numpy array
    uv is 2x(h/2)x(w/2) UV float numpy array
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def rgb2ycbcr(rgb, is_bgr=False):
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr


def ycbcr2rgb(ycbcr, is_bgr=False, clamp=True):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0., 1.)
    return rgb


def yuv_444_to_420(yuv):
    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]

    return y, _downsample(uv)
