set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging_bf16.yaml \
    data.train_files=/scratch/high_modality/geom_train_upsampled.jsonl \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=Qwen/QVQ-72B-Preview \
    trainer.n_gpus_per_node=4 \
    worker.rollout.tensor_parallel_size=4 \
    trainer.experiment_name=drpo_72b