# DATS6450-CloudComputing-DogBreedRecognition
Final group project for DATS 6450 Cloud Computing

Authors: Jesse Borg, Sarah Gates, Luis Ahumada

Files:

Dog Breed Image Recognition.pdf (Report)
DATS 6450 - CC Project Presentation.pdf (PDF presentation)
dog_breed.py (Image upload and model training)
S3 Image.py (Image upload to S3 using boto3)
The project was a machine learning project which aimed to classify different dog breeds from a number of images. The set of images was split into a training and testing set so that the program can learn which images belong to certain breeds. It will then try to classify the testing set and the quality of the program is indicated by the success rate of classification. For this project, instead of storing the images on a laptop and using downloaded programs to run the algorithms, the images will be stored online and Python will be run on an IDE (PyCharm) using AWS.

The data from this project was obtained from a predefined dataset which is included in python and is called ‘ Stanford Dogs Dataset’. This dataset was useful as it contained enough images to be able to train the program adequately, and its wide variety of dog breeds means that it will challenge the algorithm. The description of the dataset is as follows:

Images of 120 breeds of dogs from around the world
Number of categories: 120
Number of images: 20,580
Annotations: Class labels, Bounding boxes
Out of the 20,580 images, 12,000 will be used for training (58%) and 8,580 will be used for testing (42%).
The model successfully classified the test set (8,580 images) with an accuracy of 0.8480 and a validation loss of 0.8236. The images were successfully uploaded to our S3 bucket using boto3.

Deploying the project in AWS tools allowed us to successfully implement a project from start to finish using the variety of cloud services available. We were able to take a developed model and build the pipeline using tools like Boto3, S3, and EC2 with strong results. Since the dataset was curated, the next steps we could explore would be working with raw data that needs more cleaning processes. There are many ways that this project could be expanded, and we feel accomplished that the bulk of the machine learning processes are already completed.
