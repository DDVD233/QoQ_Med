set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/outputs/drpo_new_nvidia_custom_encoder_ups \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.experiment_name=drpo_custom_encoder_eval