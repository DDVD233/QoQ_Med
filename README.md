# QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training

![QoQ-Med Model Overview](images/model_training.jpeg)

This repository contains the code, model weights, and training pipeline for QoQ-Med (Qwen Omni-Reasoning on Medical Questions), a multimodal clinical foundation model with reasoning capabilities.

## Overview

QoQ-Med is the first open generalist clinical foundation model that jointly reasons across:
- Medical images (2D/3D)
- Time-series signals (ECG)
- Text reports

The model is trained with our novel Domain-aware Relative Policy Optimization (DRPO), a reinforcement learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, addressing performance imbalance in heterogeneous clinical data.

## Key Features

- **Multimodal Integration**: Processes and reasons across 1D, 2D, and 3D clinical data
- **Domain-Aware Training**: DRPO balances learning across 9 clinical domains
- **Enhanced Interpretability**: Generates reasoning traces and highlights salient regions
- **State-of-the-Art Performance**: Outperforms existing open-source clinical MLLMs

## Available Resources

- **Model Weights**: QoQ-Med-7B and QoQ-Med-32B
- **Training Pipeline**: Complete DRPO implementation
- **Reasoning Traces**: 2.61 million question-answer pairs with intermediate reasoning

## Clinical Domains

QoQ-Med spans multiple clinical specialties:
- Cardiology (ECG, Chest X-ray)
- Radiology (CT, MRI, Ultrasound)
- Dermatology
- Ophthalmology (Fundus)
- Pathology
- Mammography

## Important Note

This model is intended for research purposes only. Before extensive real-world testing (like human trials), it is not suitable for clinical deployment. This is a research preview, not a product approved by federal agencies.