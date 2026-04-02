
# MAE-XNT: A Foundation Model for Segmenting Neuronal Tissue Volumes Generated with X-Ray Nanotomography

This repository provides the code used to train MAE-XNT and to fine-tune it on the C432 and XPRESS datasets.
We provide below the commands to launch to reproduce our results.
Our experiments has been originally launched using two NVIDIA H200 GPU 141GB, 64 AMD EPYC™ 7543 CPU on Ubuntu 24.04

# Build and Run the Container

We use an apptainer container to run our experiments. The description of our container is given in container_MAE_XNT.def.
To build the container from the description file, run:

```
apptainer build container_MAE_XNT.sif container_MAE_XNT.def
```

To run the container, use:

```
apptainer shell --no-home --nv container_MAE_XNT.sif
```

# Obtaining the pre-training Dataset

Here are the publicly released volumes used to train MAE_XNT. There are several volumes per DOI. We will update this list according to the three years embargo policy of ESRF:

- 10.15151/ESRF-ES-433972376
- 10.15151/ESRF-ES-650704790
- 10.15151/ESRF-ES-514138492
- 10.15151/ESRF-ES-754325093
- 10.15151/ESRF-ES-1050393697
- 10.15151/ESRF-ES-404475765
- 10.15151/ESRF-ES-754325379

Training 3D images are the center crop of size 2000^3 of the provided volumes.

# Obtaining MAE-XNT

The code used to train MAE-XNT is provided in the MAE_pretraining folder
You can download the weights of MAE-XNT here : https://drive.google.com/file/d/1A6RbuGG6SqERgDIOmEadFkyy4l_t8Y-C/view?usp=sharing

# Model Fine-tuning for Cell Nuclei Segmentation
  
The code used to run the cell nuclei segmentation experiments is provided in the cell_nuclei_fine_tuning folder

## Download C432 dataset

To download the densely annotated volumes and the sparsely annotated ones from the C432 dataset, run the commands below in the cell_nuclei_fine_tuning folder:

```
python3 download_dense_vol.py $OUTPUT_PATH
python3 download_sparse_vol.py $OUTPUT_PATH
```

The volumes constituting the iid, ood and detection dataset can be found in the metadata directory.

## Fine-tuning 

To fine-tune MAE-XNT on the iid dataset for different amount of training data (100%, 10%, 1%), run:

```
python3 fine_tune.py metadata/iid_12train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH
python3 fine_tune.py metadata/iid_12train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH --ds_crop_size 230
python3 fine_tune.py metadata/iid_12train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH --ds_crop_size 110
```

To fine-tune MAE-XNT on the ood dataset for different amount of training data (100%, 10%, 1%), run:

```
python3 fine_tune.py metadata/ood_18train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH
python3 fine_tune.py metadata/ood_18train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH --ds_crop_size 230
python3 fine_tune.py metadata/ood_18train_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH --ds_crop_size 110
```

To fine-tune MAE-XNT on the detection dataset run:

```
python3 fine_tune.py metadata/detection_ds.json $SAVING_DIR $FOUNDATION_MODEL_PATH
```

## Training From Scratch

To train a Unetr from scratch on the iid dataset for different amount of training data (100%, 10%, 1%), run:

```
python3 train.py metadata/iid_12train_ds.json $SAVING_DIR
python3 train.py metadata/iid_12train_ds.json $SAVING_DIR --ds_crop_size 230
python3 train.py metadata/iid_12train_ds.json $SAVING_DIR --ds_crop_size 110
```

To train a Unetr from scratch on the ood dataset for different amount of training data (100%, 10%, 1%), run:

```
python3 train.py metadata/ood_18train_ds.json $SAVING_DIR
python3 train.py metadata/ood_18train_ds.json $SAVING_DIR --ds_crop_size 230
python3 train.py metadata/ood_18train_ds.json $SAVING_DIR --ds_crop_size 110
```

To train a Unetr from scratch on the detection dataset, run:

```
python3 train.py metadata/detection_ds.json $SAVING_DIR
```

## Evaluation

To measure the performance on the iid validation and test sets, run

```
python3 eval.py metadata/iid_12train_ds.json $CHECKPOINT_PATH validation --mask_border_size 50
python3 eval.py metadata/iid_12train_ds.json $CHECKPOINT_PATH test --mask_border_size 50
```

To measure the performance on the ood validation and test sets, run

```
python3 eval.py metadata/ood_18train_ds.json $CHECKPOINT_PATH validation --mask_border_size 50
python3 eval.py metadata/ood_18train_ds.json $CHECKPOINT_PATH test --mask_border_size 50
```

To measure the performance on the detection validation_sphericity, validation_early_stop and test sets, run

```
python3 get_detection_perf.py metadata/detection_ds.json $CHECKPOINT_PATH validation_early_stop --volume_threshold $VOLUME_THRESHOLD  --sphericity_threshold $SPHERICITY_THRESHOLD
python3 get_detection_perf.py metadata/detection_ds.json $CHECKPOINT_PATH validation_sphericity --volume_threshold $VOLUME_THRESHOLD  --sphericity_threshold $SPHERICITY_THRESHOLD 
python3 get_detection_perf.py metadata/detection_ds.json $CHECKPOINT_PATH test --volume_threshold $VOLUME_THRESHOLD  --sphericity_threshold $SPHERICITY_THRESHOLD 
```

# Model Fine-tuning for Axon Segmentation

The code used to run the axon segmentation experiments is provided in the axon_fine_tuning folder

## Data and models

You can download the dataset used by this experiment on https://xpress.grand-challenge.org/ and put it in your data folder (DATA_DIR).

## Training command 

You can launch a finetuning (the best hyperparameter set we found) using the following command : 

```
python3 axon_fine_tuning/train_unetr.py $DATA_DIR $SAVING_DIR --found_model_checkpoint_path $FOUNDATION_MODEL_PATH  
```

Training the model in this configuration requires 280 Gb of GPU RAM, you can adjust the batch size if needed but it will require to tune the other hyperparameters.

Launching a training from scratch with the best hyperparameters we found can be done using the following command : 

```
python3 axon_fine_tuning/train_unetr.py $DATA_DIR $SAVING_DIR --mode_train scratch --lr 5e-4 --weight_decay 0.05 --num_steps 1000 --saving_period 100
```

# Model evaluation 

To evaluate your model on the validation dataset you can use the following command :

```
python3 axon_fine_tuning/eval.py $DATA_DIR $CHECKPOINT_PATH --offset 252 --volume_size 699 --file val --threshold 0.7 --batch_size $BATCH_SIZE
```

To evaluate your model on the test dataset you can use the following command (The segmentation pipeline on the test dataset requires approximately 700 Gb of RAM memory) : 

```
python3 axon_fine_tuning/inference.py $DATA_DIR $CHECKPOINT_PATH --offset 99 --volume_size 1002 --file test --threshold 0.7 --batch_size $BATCH_SIZE
```

Then you will have to compress it in a zip file and submit it on https://xpress.grand-challenge.org/.
All the results are saved in the model folder.

# Citation

This paper has been accepted to CVPR26 Findings.

# License

This project is under MIT license




