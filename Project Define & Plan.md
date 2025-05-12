# Step 1- Definition
## Machine Learning project
Goal in this project is to build and end-to-end AI/ML deployment on mobile device, in this assignment I use CSC puhti supercomputer to run my code and train my model. After I am done with this project, I have knowledge of how to use Neural Networks in machine learning and also how to apply those in android applications.
Steps for this project is:
1.	Define & Plan
2.	Collect & Prepare Data
3.	Train & Evaluate on CSC Puhti Supercomputer
4.	Optimize for Mobile Device
5.	Build the Android App
6.	Test & Demo
 
The deadline for this project is 16 of May 2025 so the project time frame is quite short.

## Define & Plan
The application I selected is to help people to see how healthy they are eating; in my application you can take pictures using your phone’s camera and then model scan image and see what foods there is and then it tells you how healthy that food is. There is many of these kinds of applications already and many health apps are using this already, but I try eating healthily again, so with this app I can look at my eating habits and this is something that is interesting for me and practical.

**Input:** Image
	Image is taken by phone camera in application
 
**Output:** Healthy food and not healthy food
	Version 1 I have simple binary output so I just show if my model can classify if food in image is healthy or unhealthy, model is trained by using certain 	foods to do this classification.
 
**Success metric:** Accuracy
	Success metrics are how accurately model can classify if food in image is healthy or unhealthy food, later when I do more versions, I can add that it’s 	also looking nutrition and metrics
 
## Model
My pretrained model will be **MobileNetV4** because I use vision/camera in my application.
MobileNet is a type of **convolutional neural network (CNN)**, and it is designed for image classification, object detection and other computer vision tasks. MobileNet’s are designed for small sizes, and they have low power consumption, so they are good for android app. Version 4 was published in September 2024 and I am using this version in my project. It includes the 'universal inverted bottleneck,' which integrates both inverted residuals and inverted bottlenecks as special cases. This allows the model to efficiently use attention modules, including multi-query attention.

## Data
Because time frame is short for this project and I have only 9 days I decide to use readymade datasets, I have food images from http://foodcam.mobi/dataset256.html , https://www.kaggle.com/datasets/dansbecker/food-101 and https://www.kaggle.com/datasets/trolukovich/food11-image-dataset.
For this project I combine these datasets to classify if food is healthy or not healthy, I combine these three datasets and add some images that I collected from google to support healthy food images. This way I get a large dataset to train my model. Later I can then make my model better in my own time and even classify how many calories food have and maybe my model can tell what ingredients certain foods have if I give image from pasta dish.

I have download also CSV files where there is food names that match images and nutrition information from different foods and ingredients but in this point, I don’t use those yet and I keep them for later use.

## GitHub
I use GitHub to record my progress, and I save there my documentary and code so it can be my portfolio from this project. This is the first time I have used GitHub like this, so I get training in that too when project documentary is done.
