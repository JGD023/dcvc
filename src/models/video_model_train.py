# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math 

import torch
import torch.nn.functional as F
from torch import nn

from .common_model_train import CompressionModel
from ..layers.layers_train import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8

qp_shift = [0, 8, 4]
# extra_qp = max(qp_shift)
extra_qp = 0

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        devive = x1.device
        quant = quant.to(devive)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        device = feature.device
        quant_step = quant_step.to(device)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        device = feature.device
        quant_step = quant_step.to(device)
        feature = feature * quant_step
        return feature


class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        out = self.conv(x)
        device = out.device
        quant_step = quant_step.to(device)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None


class DMC(CompressionModel):
    def __init__(self):
        super().__init__(z_channel=g_ch_z, extra_qp=extra_qp)
        self.qp_shift = qp_shift

        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.ModuleList([nn.Conv2d(g_ch_d, g_ch_d, 1) for _ in range(3)])
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        # self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        # self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        # self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        # self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1)))

        self.q_encoder = torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1))
        self.q_decoder = torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1))
        self.q_feature = torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1))
        self.q_recon = torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1))

        self.previous_frame_recon = None
        self.previous_frame_feature = None

        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0


    def load_dict(self, pretrained_dict, strict=True):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict, strict=strict)
        
    def reset_ref_feature(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def add_ref_frame(self, feature=None, frame=None, increase_poc=True):
        ref_frame = RefFrame()
        ref_frame.poc = self.curr_poc
        ref_frame.frame = frame
        ref_frame.feature = feature
        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        self.dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def apply_feature_adaptor(self):
        if self.dpb[0].feature is None:
            return self.feature_adaptor_i(F.pixel_unshuffle(self.dpb[0].frame, 8))
        return self.feature_adaptor_p(self.dpb[0].feature)

    def feature_adaptor_my(self, ref, feature, index):
        if feature is None:
            feature = self.feature_adaptor_i(F.pixel_unshuffle(ref, 8))
        else:
            feature = self.feature_adaptor_p[index](feature)
        return feature

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def prepare_feature_adaptor_i(self, last_qp):
        if self.dpb[0].frame is None:
            q_recon = self.q_recon
            self.dpb[0].frame = self.recon_generation_net(self.dpb[0].feature, q_recon).clamp_(0, 1)
            self.reset_ref_feature()

    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        index = torch.tensor(0, dtype=torch.int).to(z.device)
        prob = bit_estimator(z + 0.5, index) - bit_estimator(z - 0.5, index)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def get_mse(self, x, x_hat, Pyuv420=False):
        if Pyuv420:
            org_y, org_u, org_v = yuv_444_to_420(x)
            rec_y, rec_u, rec_v = yuv_444_to_420(x_hat)
            mse_y = torch.mean((org_y-rec_y).pow(2))
            mse_u = torch.mean((org_u-rec_u).pow(2))
            mse_v = torch.mean((org_v-rec_v).pow(2))
            mse = (4 * mse_y + mse_u + mse_v) / 6
        else:
            mse = torch.mean((x-x_hat).pow(2))
        return mse

    def forward_one_frame(self, x, ref_frame, ref_feature, fa_idx, Pyuv420=False):
        q_encoder = self.q_encoder.to(x.device)
        q_decoder = self.q_decoder.to(x.device)
        q_feature = self.q_feature.to(x.device)
        q_recon = self.q_recon.to(x.device)
        x_yuv = rgb2ycbcr(x)

        feature = self.feature_adaptor_my(ref_frame, ref_feature, fa_idx)
        # print(1111111111111)
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        # ctx_0 = torch.zeros_like(ctx).to(ctx.device)
        y = self.encoder(x_yuv, ctx, q_encoder)

        # hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(y)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_res, y_q, y_hat, scales_hat = \
            self.forward_prior_2x(y, params, self.y_spatial_prior)
        x_hat_ycbcr, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        x_hat = ycbcr2rgb(x_hat_ycbcr)
        mse_loss = self.get_mse(x_hat, x, Pyuv420)

        if self.training:
            y_for_bit = self.add_noise(y_res)
            z_for_bit = self.add_noise(z)
        else:
            y_for_bit = y_q
            z_for_bit = z_hat
        im_shape = x.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]

        total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
        total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)

        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bit = total_bits_y + total_bits_z
        
        # bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        # bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, torch.tensor(0, dtype=torch.int).to(z.device))

        # bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        # bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z
        # bit = torch.sum(bpp) * pixel_num

        return {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "mse_loss": mse_loss,
                "recon_image": x_hat_ycbcr,
                "feature": feature,
                "bit": bit,
                }

    def inference(self, x, fa_idx):
        q_encoder = self.q_encoder.to(x.device)
        q_decoder = self.q_decoder.to(x.device)
        q_feature = self.q_feature.to(x.device)
        q_recon = self.q_recon.to(x.device)

        feature = self.feature_adaptor_my(self.dpb[0].frame, self.dpb[0].feature, fa_idx)
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        # ctx_0 = torch.zeros_like(ctx).to(ctx.device)
        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_res, y_q, y_hat, scales_hat = \
            self.forward_prior_2x(y, params, self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)
        self.add_ref_frame(feature, x_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, torch.tensor(0, dtype=torch.int).to(z.device))

        im_shape = x.size()
        pixel_num = im_shape[2] * im_shape[3]

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z
        bit = torch.sum(bpp) * pixel_num

        return {"bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": x_hat,
                "bit": bit.item(),
                }

    def compress(self, x, fa_idx):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        device = x.device
        q_encoder = self.q_encoder
        q_decoder = self.q_decoder
        q_feature = self.q_feature
        
        feature = self.feature_adaptor_my(self.dpb[0].frame, self.dpb[0].feature, fa_idx)
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)
        cuda_event_z_ready = torch.cuda.Event()
        cuda_event_z_ready.record()
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = \
            self.compress_prior_2x(y, params, self.y_spatial_prior)

        cuda_event_y_ready = torch.cuda.Event()
        cuda_event_y_ready.record()
        feature = self.decoder(y_hat, ctx, q_decoder)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        
        with torch.cuda.stream(cuda_stream):
            self.entropy_coder.reset()
            cuda_event_z_ready.wait()
            self.bit_estimator_z.encode_z(z_hat_write, 0)
            cuda_event_y_ready.wait()
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.entropy_coder.flush()
        
        bit_stream = self.entropy_coder.get_encoded_stream()
        
        torch.cuda.synchronize(device=device)
        self.add_ref_frame(feature, None)
        
        return {
            'bit_stream': bit_stream,
        }

    def decompress(self, bit_stream, sps):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_decoder = self.q_decoder
        q_feature = self.q_feature
        q_recon = self.q_recon

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        self.bit_estimator_z.decode_z(z_size, 0)

        feature = self.feature_adaptor_my(self.dpb[0].frame, self.dpb[0].feature, sps['fa_idx'])
        c1, ctx_t = self.feature_extractor.forward_part1(feature, q_feature)

        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        infos = self.decompress_prior_2x_part1(params)

        ctx = self.feature_extractor.forward_part2(c1)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            y_hat = self.decompress_prior_2x_part2(params, self.y_spatial_prior, infos)
            cuda_event = torch.cuda.Event()
            cuda_event.record()

        cuda_event.wait()
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        self.add_ref_frame(feature, x_hat)
        return {
            'x_hat': x_hat,
        }

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]

    @staticmethod
    def get_rd_info(result):
        rd = {
            "bpp_y": torch.mean(result['bpp_y']),
            "bpp_z": torch.mean(result['bpp_z']),
            "bpp": torch.mean(result['bpp']),
            "mse_loss": torch.mean(result['mse_loss']),
        }
        return rd

    @staticmethod
    def get_loss_info(rd):
        info = {
            "bpp_y": rd['bpp_y'].item(),
            "bpp_z": rd['bpp_z'].item(),
            "mse_loss": rd['mse_loss'].item(),
        }
        return info

    def forward(self, x, ref_frame, ref_feature, fa_idx, loss_func=None, Pyuv420=False):
        if loss_func is None:
            return self.forward_one_frame(x, ref_frame, ref_feature, fa_idx, Pyuv420=Pyuv420)

        frame_nums = x.shape[1] // 3
        if frame_nums == 1:
            if ref_frame is None:
                # no reference frame is passed from caller, use the buffered one
                ref_frame = self.previous_frame_recon
                ref_feature = self.previous_frame_feature
            else:
                # this is the first P frame, use the reference frame from caller
                ref_feature = None
            result = self.forward_one_frame(x, ref_frame, ref_feature, fa_idx, Pyuv420=Pyuv420)
            rd = self.get_rd_info(result)
            loss = loss_func(rd,fa_idx)
            self.previous_frame_recon = result["recon_image"].detach()
            self.previous_frame_feature = result["feature"].detach()
            info = self.get_loss_info(rd)
            return loss, info

        losses = []
        ref_feature = None
        for frame_index in range(frame_nums):
            cur_frame = x[:, frame_index * 3:(frame_index + 1) * 3, :, :]
            index_map = [0, 1, 0, 2]
            fa_idx = index_map[(frame_index+1) % 4]
            result = self.forward_one_frame(cur_frame, ref_frame, ref_feature, fa_idx, Pyuv420=Pyuv420)
            rd = self.get_rd_info(result)
            loss = loss_func(rd,fa_idx)
            losses.append(loss)
            ref_frame = result["recon_image"]
            ref_feature = result["feature"]
        loss = torch.mean(torch.stack(losses))
        info = self.get_loss_info(rd)
        return loss, info

class VideoDataParallel(nn.parallel.DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rec_only(self, x, qp):
        return self.module.get_rec_only(x, qp)

    def get_rd_info(self, result):
        return self.module.get_rd_info(result)
    
    def set_noise_level(self, noise_level):
        return self.module.set_noise_level(noise_level)

    def get_noise_level(self):
        return self.module.get_noise_level()