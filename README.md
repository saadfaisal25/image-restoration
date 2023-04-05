# image-restoration
This project aims to restore images degraded by rain, blur, and noise using various restoration techniques.

## Deraining
This section utilizes deep learning to restore images degraded by rain. The pretrained model `model.pth` was trained on the Rain1800 dataset.

The dataset should be stored in a folder named `data` in the root directory. The folder should contain two subfolders `test` and `train`, each of which contain `input` and `target` folders. The `input` folder contains the input images and the `target` folder contains the corresponding ground truth images.

To install dependencies:
`pip install -r requirements.txt`

To train the model:
`python train.py`

To test the model:
`python test.py`

To test a single image:
`python single_test.py --input <path to image>`