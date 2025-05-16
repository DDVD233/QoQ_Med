set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/unified_train.json \
    data.val_files=/home/jovyan/workspace/high_modality/unified_valid_mini.jsonl \
    algorithm.adv_estimator=rloo \
    worker.actor.model.model_path=/home/jovyan/workspace/drpo_tsonly/time_series_qwen2_5_vl \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.experiment_name=rloo_vanilla_unified_new