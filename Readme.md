# Use Unet to do still image optical flow task

## Dataset
Use the Sintel Final dataset to train the model. The dataset should be structured as follows:

├──data  
&ensp;&ensp;├── images (located at Sintel/training/final)  
&ensp;&ensp;├── flows (located at Sintel/training/flows)  
&ensp;&ensp;├── KITTI
&ensp;&ensp;&ensp;&ensp;├── flow_occ (KITTI's optical flow GT)
&ensp;&ensp;&ensp;&ensp;├── image_2 (KITTI's training data)


## Training Phase
Use the following code to train the model:

```bash
python train.py
```

## Finetuning Phase
Use the following code to finetuning the model on KITTI dataset:

```bash
python finetuning.py
```

## Predict
Use the following code to predict optical flow:
```bash
python predict.py --model model_path(in checkpoints folder) --input input_image_path --output output.png

```
