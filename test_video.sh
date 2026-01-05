 python test_video_single_rate.py \
 --model_path_i ./ckpts/cvpr2025_image.pth.tar \
 --model_path_p /data/wangf/research/nvc/dcvc_rt_retry/DCVC/video_ckpts_36/checkpoint_video_ckpts_36__epo_28.pth \
 --rate_num 1 --test_config ./dataset_config_example_rgb.json \
 --cuda 1 --worker 4 --write_stream 1 --float16 False \
 --output_path ./output_png_qp36.json --force_intra_period 32 \
 --reset_interval 100 --force_frame_num 96 --check_existing 0 --verbose 0 \
 --cuda_idx 1 2 --save_decoded_frame 0 --qp_i 42
