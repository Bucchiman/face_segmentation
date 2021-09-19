cd src


# gpu
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=3 main.py --epochs 20 &
