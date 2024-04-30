now=$(date +"%Y%m%d_%H%M%S")

MODEL=CLIP_B16
KEY=CLIP_openai/cifar10/FT100_6E4_D06_NC/base

OUTPUT_DIR=OUTPUT/CLIP_ft/${KEY}/${now}
LOD_DIR=LOGDIR/CLIP_ft/${KEY}/${now}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

DATA_PATH=DATA/
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=40401 run_class_finetuning.py \
    --model ${MODEL} --data cifar10 --data_path $DATA_PATH \
    --input_size 224 \
    --finetune True \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} --log_dir ${LOD_DIR} \
    --batch_size 64 --lr 6e-4 --update_freq 1 \
    --warmup_epochs 10 --epochs 100 \
    --layer_decay 0.6 \
    --drop_path 0 \
    --dist_eval --eval_all --save_ckpt --save_ckpt_freq 50 \
    --enable_deepspeed \
    --layer_scale_init_value 0 \
    --abs_pos_emb --disable_rel_pos_bias \
    --weight_decay 0.05 --mixup 0 --cutmix 0 \
    --nb_classes 10 --model_prefix visual.\
    --model_ema --model_ema_decay 0.9998 \
    2>&1 | tee -a ${OUTPUT_DIR}/log.txt


