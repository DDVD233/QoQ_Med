set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/outputs/drpo_new_nvidia_custom_encoder_ups \
    trainer.n_gpus_per_node=8 \
    trainer.experiment_name=drpo_new_nvidia_vanilla