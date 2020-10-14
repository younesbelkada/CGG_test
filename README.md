# CGG_test

Implementation of my solution for the CGG Machine Learning assessment. I decided to use PyTorch and Python 3.7 to implement my solution. 

## Requirements

To be able to run the training script, please make sure that you have installed the following libraries :
```
torch == 1.5.1+cu101
opencv-python
```
Install the required packages using:
```
pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install opencv-python
pip3 install scikit-learn
```

## Code explanation

The repo is divided into 3 codes:
* ```data.py``` : A file that contains the definition of out data structure. It defines how we load the data and preprocess it.
* ```network.py``` : A file that defines the network for the task.
* ```utils.py``` : A file that contains the utility functions for evaluating our model.
* ```main.py``` : The main script that has to run to be able to see some results.

### Input data

The data files are in ```.tif``` format. It's a special format since the pictures are encoded in 16-bit so it requires so preprocessing to be able to load them. It's better to normalize the input using an in-build function that comes with OpenCV (line 30 at ```data.py```):
```{python}
(cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX).astype(np.float32))
```
After calling this function the picture will be normalized and the pixel values will have a range betweeb 0 and 1 (that corresponds to the grayscale instensity of the picture). One may also change this input preprocessing pipeline to have a better accuracy.

### Output Data

Since the output masks are also in a ```.tif``` format, we need to change some values. The details are provided in the line 31 at ```data.py```.

### Network

The network architecture is highly inspired from the U-Net architecture defined in this ![paper](url = https://www.nature.com/articles/s41598-019-53797-9) and defined in the file ```network.py```. One may also change the hyperparameters of this architecture to try to see if we can achieve a better accuracy. I also decided to use the CrossEntropyLoss for this task.

#### Discussions around the last layer

I decided to add a Sofmax function that is going to be applied through all the output channels (recall that output_channels = nb_classes). Some articles on the internet suggests to not go with this approach, one may also tune that.

### Evaluation metrics

For binary classification, I decided to compute the accuracy per class. This can be done by computing the confusion matrix usning ```sklearn```.  The accuracy per class is defined by : nb_TruePositive/nb_InstanceClass with nb_TruePositive defined as the number of True Positives in the given class and nb_InstanceClass defined as the number of the instances of the corresponding class. One can also compute the IoU (Intersection Over Union), but given the very low distribution of the class 1, it could be better to visualize the IoU which makes more sense in our case. The IoU is implemented in this version but not implemented in the multi-class classification

For the multi class classification I decided to compute the average accuracy. But one may compute the IoU per class and the mean IoU due to the very high distribution of the class 0 amongst the other classes. 

### Training and evaluating

To be able to train and evaluate the model run the following regarding what you want to do:

#### Binary classification

Just run 
```
python3 main.py train --lr [LEARNING_RATE] --bs [BATCH_SIZE] --epochs [EPOCHS] --input_dir [INPUT_DIR] --output_dir [MASKS_DIR]
```

#### Multi-class classification
```
python3 main.py train --lr [LEARNING_RATE] --bs [BATCH_SIZE] --epochs [EPOCHS] --input_dir [INPUT_DIR] --output_dir [MASKS_DIR] --multi_class True
```

