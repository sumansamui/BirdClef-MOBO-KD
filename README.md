# BirdClef-MOBO-KD
This repository contains the code, experimental setup, and supplementary materials for the paper:

**"Optimizing TinyML Models for Bird Call Recognition via Multi-Objective Bayesian Search and Knowledge Distillation"**


# Abstract
In this work, we propose a hardware-efficient framework for automated bird call identification, tailored to the stringent resource constraints of TinyML deployment. The approach follows a two-stage pipeline that integrates multi-objective Bayesian optimization and knowledge distillation, followed by post-training quantization. In the first stage, Pareto-optimal deep learning architectures—custom residual networks—are discovered under an approximate 2 MB size constraint, effectively balancing classification accuracy and resource efficiency. These compact student models are then refined through distillation from a high-capacity teacher model to enhance performance. Evaluated on the BirdCLEF 2021 dataset, the student models show an average improvement of 3.5% in test accuracy after distillation, with the smallest model gaining up to 6.8%. The final quantized (INT8) models are deployable on TinyML hardware, confirming the feasibility of real-time, on-device bird call monitoring for large-scale ecological applications.



