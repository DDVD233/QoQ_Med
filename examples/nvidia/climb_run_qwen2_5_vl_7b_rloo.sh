set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train_upsampled.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=rloo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    worker.rollout.tensor_parallel_size=2 \
    trainer.experiment_name=rloo_7b_vanilla