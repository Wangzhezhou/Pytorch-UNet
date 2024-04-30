# Use Unet to do still image optical flow task

## Dataset
Use the Sintel Final dataset to train the model. The dataset should be structured as follows:

data/
├── images (located at Sintel/training/final)
└── flows (located at Sintel/training/flows)

## Training Phase
Use the following code to train the model:

```bash
python train.py
```


## Predict
Use the following code to predict optical flow:
```bash
python predict.py --model model_path(in checkpoints folder) --input input_image_path --output output.png

```
