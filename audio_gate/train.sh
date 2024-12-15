# # 使用默认参数训练
# python train.py

# 自定义参数训练
python train.py \
    --num_experts 0 \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-6 \
    --model-dim 512 \
    --num-layers 3 \
    --output-dir outputs_nonmoe

