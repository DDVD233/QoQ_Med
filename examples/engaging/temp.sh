set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/dvdai/MedR1/VQA_Disease_Diagnosis \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node=4
