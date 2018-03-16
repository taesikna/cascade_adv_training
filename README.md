# Cascade Adversarial Training Regularized with a Unified Embedding


This repository is created to reproduce the results in the paper:<br>
“[Cascade Adversarial Machine Learning Regularized with a Unified Embedding](https://arxiv.org/pdf/1708.02582.pdf),”<br>
Taesik Na, Jong Hwan Ko and Saibal Mukhopadhyay,
International Conference on Learning Representations (ICLR), Apr 2018

Please consider to cite our paper in your publications if it helps your research.

## Requirements

The code was tested with Python 2.7.12 and Tensorflow 1.4.0.

## Running

### Initial Training
First, we train 20-layer ResNet with CIFAR10 dataset for 10 epochs.
It will automatically download CIFAR10 dataset under ./data directory.
And the trained model will be stored under ./ref/resnet20_cifar10.
```
python main.py --max_epochs=10 \
               --is_train=True \
               --adversarial=False \
               --train_dir=./ref/resnet20_cifar10
```

### Adversarial Training (Optional)
If you want to perform vanilla adversarial training, please use the below command.
```
python main.py --is_train=True \
               --adversarial=True \
               --restore=True \
               --checkpoint_dir=./ref/resnet20_cifar10 \
               --train_dir=./checkpoint/resnet20_cifar10_adv
```

### Adversarial Training with Pivot Loss
Next, we train a network with pivot loss. The trained model will be stored in ./checkpoint/resnet20_cifar10_pivot
```
python main.py --is_train=True \
               --adversarial=True \
               --pivot_loss_factor=0.0001 \
               --restore=True \
               --checkpoint_dir=./ref/resnet20_cifar10 \
               --train_dir=./checkpoint/resnet20_cifar10_pivot
```

### Cascade Adversarial Training with Pivot Loss
Now, we perform cascade adversarial training with pivot loss.
First, we craft iter_fgsm images from the already defended network which is stored in ./checkpoint/resnet20_cifar10_pivot.
And we train a network with adversarial images from the network being trained and stored iter_fgsm images.

```
python main.py --is_train=False \
               --restore_inplace=True \
               --save_iter_fgsm_images=True \
               --test_data_from=train \
               --train_dir=./checkpoint/resnet20_cifar10_pivot
python main.py --is_train=True \
               --adversarial=True \
               --pivot_loss_factor=0.0001 \
               --cascade=True \
               --saved_iter_fgsm_dir=./checkpoint/resnet20_cifar10_pivot \
               --restore=True \
               --checkpoint_dir=./ref/resnet20_cifar10 \
               --train_dir=./checkpoint/resnet20_cifar10_pivot_cascade
```

### Ensemble Adversarial Training with Pivot Loss
If you want to perform ensemble adversarial training with pivot loss, please use the following series of commands.
It will train vanilla 20-layer ResNet and 110-layer ResNet, and those networks will be used as source networks for ensemble adversarial training.
```
python main.py --is_train=True \
               --adversarial=False \
               --train_dir=./checkpoint/r20_ens_source
python main.py --is_train=True \
               --adversarial=False \
               --resnet_n=18 \
               --train_dir=./checkpoint/r110_ens_source
python tf_rename_variables.py --checkpoint_dir=checkpoint/r20_ens_source \
                              --replace_from=main \
                              --replace_to=r20_ens_source
python tf_rename_variables.py --checkpoint_dir=checkpoint/r110_ens_source \
                              --replace_from=main \
                              --replace_to=r110_ens_source
python main.py --is_train=True \
               --adversarial=True \
               --pivot_loss_factor=0.0001 \
               --ensemble=True \
               --restore=True \
               --checkpoint_dir=./ref/resnet20_cifar10 \
               --train_dir=./checkpoint/resnet20_cifar10_pivot_ensemble
```

### Black Box Attack Test
For black box test, we first save the adversarial images and test those images for the target network.<br>
The following series of commands will save the adversarial images (step_ll, step_fgsm, step_rand, iter_ll, iter_fgsm) from the source network (adversially trained network with pivot loss) with epsilons (2, 4, 8, 16), and test accuracy for the target network (a network trained with cascade adversarial training with pivot loss).

```
python main.py --is_train=False \
               --restore_inplace=True \
               --save_adver_images=True \
               --test_data_from=validation \
               --train_dir=./checkpoint/resnet20_cifar10_pivot
python main.py --is_train=False \
               --restore_inplace=True \
               --use_saved_images=True \
               --saved_data_dir=./checkpoint/resnet20_cifar10_pivot \
               --train_dir=./checkpoint/resnet20_cifar10_pivot_cascade
```

## Contact

Feel free to contact me at taesik.na@gatech.edu

