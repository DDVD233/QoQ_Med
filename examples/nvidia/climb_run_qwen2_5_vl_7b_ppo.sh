set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/home/jovyan/workspace/high_modality/geom_train.jsonl \
    data.val_files=/home/jovyan/workspace/high_modality/geom_valid_mini.jsonl \
    algorithm.adv_estimator=gae \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    worker.critic.model.model_path=Qwen/Qwen2.5-7B-Instruct \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    trainer.n_gpus_per_node=8 \
    worker.rollout.tensor_parallel_size=2 \
    trainer.experiment_name=ppo_7b_vanilla