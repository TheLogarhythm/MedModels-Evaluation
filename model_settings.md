### MedMamba

- **MedMNIST**, we followed the same training settings of MedMNISTv2 and MedViT without making any modifications to the original settings. Concretely, we trained all of the MedViT variants (MedMamba-T, MedMamba-S, and MedMamba-B) for *100 epochs* and used a *batch size of 128*. We employed an AdamW optimizer with an *initial learning rate of 0.001*, the learning rate is *decayed by a factor set of 0.1 in 50 and 75 epochs*.
- **Non-MedMNIST**(the dataset that does not belong to MedMNIST),we employed the **AdamW** optimizer with a *0.0001 initial learning rate,B1 of 0.9,B2 of 0.999,and weight decay of 1e-4* and Cross-Entropy Loss to optimize the model parameters. We trained each model for *150 epochs* and used a *batch size of 64*.We used an *early-stop strategy* to prevent model overfitting. Besides,we did not apply any data augmentation strategy and pre-training to demonstrate as much as possible that the results of all model metrics benefit from MedMamba’s unique architecture.

### MedViT

- **MedMNIST dataset**: follow the same training settings of the MedMNISTv2 without making any changes from the original settings. Specifically, we train all of the MedViT variants for *100 epochs* on NVIDIA 2080Ti GPUs, and use a *batch size of 128*. The images are first resized to a size of 224 x 224 pixels. We employ an *AdamW optimizer with an initial learning rate of 0.001*, the learning rate is *decayed by a factor set of 0.1 in 50 and 75 epochs*.

### MedViTV2

- **MedMNIST**: adhered to the same training configurations as MedMNISTv2 and MedViTV1, without modifying the original settings. Specifically, all MedViT variants were trained for *100 epochs* on an NVIDIA A100 GPU with 40 GB of VRAM, using a *batch size of 128*. The images were resized to 224 x 224 pixels. We used the *AdamW optimizer* with an *initial learning rate of 0.001, which was decayed by a factor of 0.1 at the 50th and 75th epochs*.
- **Non-MedMNIST**: During the training of the NonMNIST datasets (PAD-UFES-20, Fetal-Planes-DB, CPN X-ray, ISIC2018, and Kvasir), we adhered strictly to the training configurations outlined in Medmamba. The MedViT variants underwent training for *150 epochs*, utilizing a *batch size of 64*. Images were resized to 224 x 224 pixels. Furthermore, we utilized the *AdamW optimizer*, setting the *initial learning rate at 0.0001, with B1 at 0.9, B2 at 0.999, and a weight decay of 1e-4*. Cross-Entropy Loss was employed to optimize the model parameters.

## Try

- **Non-MedMNIST**
    - *beta 1*: 0.9, *beta 2*: 0.999 (AdamW defaults)
    - *early-stop* (Not use for first time)
    - *epochs*: 150
    - *weight decay*: 1e-4
    - *learning rate*: (0.0001, 0.0002, 0.0004)
    - *batch size*: 64 in original papers