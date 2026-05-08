CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /shared/models/Qwen/Qwen3-0.6B \
    --tuner_type lora \
    --dataset /data/heyudong/dataset/sft/qwen3_32b_distill_1k_messages.jsonl \
              /data/heyudong/dataset/sft/self_cognition_messages.jsonl \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir model/qwen3-sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4