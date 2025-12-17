# IR-VIS

- More elegant solution to curriculum learning
    - Modify train(config, start_epoch) function to replace train() for handling training from checkpoints
    - train_curriculum() just repeatedly calls train(stage_config, start_epoch) for each stage
    - Curriculum object method that can return proper config given an epoch
- Train on multiple resolutions?
- Train on different resolutions between IR and VIS?
- Train on higher-order model and then prune during training?
- Training on multiple GPUs across multiple nodes
- Literature review and related work section
- Find potential sponsors or collaborators?
    - Anduril
    - FLIR
    - NGA
