python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9814 \
./run.py \
--config ./config/vast/11777/mix-120-inference.json \
--pretrain_dir './output/vast/pretrain_vast' \
--output_dir './output/inference/' \
--test_batch_size 128 \
--generate_nums 3 \
--captioner_mode true \
--mode 'testing' \
--checkpoint /opt/dlami/nvme/VAST/output/vast/pretrain_vast/ckpt/model_step_204994.pt


#### CHANGGE: config, pretrain_dir, output_dir