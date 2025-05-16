set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train_upsampled.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=grpo \
    worker.actor.model.model_path=/home/jovyan/workspace/qwen25_vision_model \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=grpo_custom_encoder_ups