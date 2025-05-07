# 1.	Define & Pla
The application I selected is to help people to see how healthy they are eating; in my application you can take pictures using your phone’s camera and then
model scan image and see what foods there is and then it tells you how healthy that food is. There is many of these kinds of applications already and many
health apps are using this already, but I try eating healthy again, so with this app I can look my eating habits and also this is something that is
interesting for me and also practical.

- **Input:** Image
	Image is taken by phone camera in application
-**Output:** Food nutrition – Healthy food and not healthy food
Version 1 I have simple binary output so I just show if my model can classify if food in image is healthy when nutrition’s are certain and unhealthy if they are not
- **Success metric:** Accuracy
	Success metrics are how accurately model can classify if food in image is healthy or unhealthy food, later when I do more versions, I can add that
  it’s also looking nutrition and metrics

**Model**
My pretrained model will be MobileNetV4 because I use vision/camera in my application.

MobileNet is a type of convolutional neural network (CNN), and it is designed for image classification, object detection and other computer vision tasks.
MobileNet’s are designed for small sizes, and they have low power consumption, so they are good for android app. Version 4 was published in September 2024 and
I am using this version in my project. It includes the 'universal inverted bottleneck,' which integrates both inverted residuals and inverted bottlenecks as
special cases. This allows the model to efficiently use attention modules, including multi-query attention.

**Data**
Because time frame is short for this project and I have only 9 days I decide to use readymade datasets, I have images from 256-kind food datase and I have
around 31 395 images from different foods. This way I collect my image dataset fast and I get good quality images. I also have txt file that tells category and
foods Japanese name and English name. This dataset can be used freely if it’s non-commercial research purpose and this project is like that for me.
Dataset link: http://foodcam.mobi/dataset256.html 

My other dataset is CSV file that contains 555 rows of nutrition data from different foods like pesto, white beans, kimchi etc. License for this dataset is under
creative commons V4.0  and it’s free to use. Dataset link: https://github.com/google-research-datasets/Nutrition5k 

For this project I combine these datasets to classify if food is healthy or not healthy, I didn’t want to use the same images that
are with nutrition data because I want little more challenge when I didn’t collect my images by myself, so I need now do more data cleaning when
nutrition data and images are not from same place.

