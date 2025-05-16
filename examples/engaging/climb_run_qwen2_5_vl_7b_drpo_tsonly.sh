set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.train_files=/scratch/high_modality/multimodal/mimiciv/annotation_train.jsonl \
    data.val_files=/scratch/high_modality/multimodal/mimiciv/annotation_valid.jsonl \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/outputs/time_series_qwen2_5_vl \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.experiment_name=drpo_custom_encoder_eval