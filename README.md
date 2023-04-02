# Face-Mask-Detection

# who to run the script

* Download the appropriate Python 3.10.1 package from Python's official website : https://www.python.org/downloads/.

* Open the downloaded folder and double-click the python-3.10.1.exe executable file to start the installation process.

* Follow the instructions displayed by the installer to complete the installation process.

* Once the installation is complete, open the folder which contains the required Python packages in the terminal. Then install them using the command pip install -r requirements.txt or pip3 install -r requirements.txt.

* Ensure that there is a folder called "Images" within the project folder containing two subfolders named "train" and "test". The model should be able to train on this dataset of images located within the "Images" folder.   
If the "Images" folder is not present in the project folder, you can download it using the link provided bellow in dataset section. This folder contains two folders named train and test which contain the images that will be used to train the model.
   

* To run the model, type either python Face_Mask_Detection_model.py or
 python3 Face_Mask_Detection_model.py in the terminal. After running it, you should get a model.pth containing all the weights and necessary information for the face mask detection model. The model can be used anytime within any application or script.



# The approach
The approach taken to arrive at a model for object classification was to use the VGG16 pre-trained model. This was chosen due to the widely accepted performance of VGG16 on image-based tasks and its ability to classify images accurately with much fewer data samples than custom-built architectures. The justification for using the pre-trained model is that it is a well-established network with proven performance as compared to a custom architecture, and thus saves time and effort in getting a reliable model for the task at hand.

The results obtained from the training and testing of this model are shown in the figure result.png which is in the folder. It can be seen that for the 3 epochs used for training, the accuracy achieved by the model is calibrated around the 98% range for both the train and test datasets, respectively. This proves that the model is able to generalize to new unseen data as evidenced by the fairly close calibration of the train and test datasets performs within similar ranges.

For future enhancements, hyperparameter tuning can be done to this pre-trained model to further increase its accuracy. Additionally, other models/networks such as ResNet or AlexNet can also be tested to compare their performance on this dataset and to get the best model suitable for the task.

# Model  that i used (pre-trained models) and dataset 
To use VGG16, you can follow these steps:

Download the VGG16 pre-trained model from this link: 
https://pytorch.org/hub/pytorch_vision_vgg/
Read up on the model structure of VGG16 at this link:
https://medium.com/mlearning-ai/an-overview-of-vgg16-and-nin-models-96e4bf398484
Use the VGG16 model for your project which may involve loading weights and setting up the model in the same way as described in the link.
You can download the pre-trained weights for the VGG16 model from the link provided by running the following line of code in a terminal:
wget https://download.pytorch.org/models/vgg16-397923af.pth

# dataset
Take a look at this link:
https://github.com/JovaniPink/mask-data/tree/main/data/processed
It contains the data set i used.
