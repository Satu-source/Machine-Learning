# Step 5 - Build the Android App

In step 5 I need to build an **android app** that uses a camera so I can take new images that my model can then classify if they are healthy foods or unhealthy. I have never done any android app or any other apps, so everything was new for me. in *figure 1* is seen steps I needed to do for my project.

![image](https://github.com/user-attachments/assets/e8539625-0050-45a9-964c-973d9b70c199)

*Figure 1 Steps to make app in Android Studio*

## Step 1 Install Android Studio

This is that I need to install an android studio in my laptop

**On your laptop:**

1. Go to: https://developer.android.com/studio
2. Download Android Studio (Stable) for your OS
3. Install it and launch

**Android Studio** is the main tool **developers use to make Android apps** — like Visual Studio Code, but for Android. This part was easiest to do because download and installation was easy, and I just take button next in installing.

## Step 2 Creating new App

This step happens inside Android studio, and this is when I do my new app. I want to make simple app because this is my first app ever, so I don’t want to start with it too complicated.

Below is the steps I need to do to make an app.

1. In Android Studio → click "New Project"

   ![image](https://github.com/user-attachments/assets/1b3d3cb6-a59b-4b1d-bd50-59c9a6eaa663)

3. Choose "Empty Activity"

   ![image](https://github.com/user-attachments/assets/f0f7a362-2d69-4a0b-b486-07bed87e8b34)

5. Give name, programming language, minimum SDK

Name: FoodHealtClassifier
Language: Kotlin
Minimum SDK: API 24 (Android 7.0) or higher
Finish - > Android Studio sets up your app

  ![image](https://github.com/user-attachments/assets/0172500f-f95e-4a5f-b972-2cacda70f88b)

## Step 3 PyTorch model 

I use **PyTorch** to build my app, there is also a chance to use TensorFlow but when I try make model for that and run it my accuracy was only 50% so then I just stay at PyTorch model. So, I had already had my *model_optimized.plt* from **step 4** so then I just started building my model. I need to use lots of **Gemini AI** in this part because I haven’t built any android app before, and the time frame was short, so I need to take shortcut.

**Basic steps to build my model were:**

1. Exporting your PyTorch model to TorchScript format.
2. Adding the PyTorch Mobile dependency to your Android project.
3. Loading the model in your Android app.
4. Preprocessing the input image (from the camera).
5. Running inference with the model.
6. Post-processing the output to get the classification (healthy/unhealthy).
7. Displaying the result

First was to convert your trained PyTorch model (*.pt file*) into the TorchScript format (.ptl for mobile). This is done in Python environment where I trained the model. I did this in spyder locally to save time in step 4.

command that I give inside **Gemini** was: *“I need to make code that is using my CNN model with PyTorch and now i need code for that, can you give me first basic code and where i need to copy paste that. My model is classifying food if it's healthy or unhealthy and then my android app take picture and model classify if that picture include healthy food or unhealthy food”*

With this command Gemini give me basic code, the code doesn’t work right away, and I need to make lots of changes and many versions when I try to get my app work. 

I change codes inside different files like MainActivity and Colors.xml and I also made some new files

**.kt files**

- **MainActivity.kt** is my app’s main screen logic, it runs first when app launches and handles UI, button clicks and camera/photo logic etc.
- **FoodClassifier.kt** is my model’s loader and classifier, it loads my model_optimized.plt file and returns prediction results.

For **UI, colors and layout** I had their own files, and I need to edit those a lot also to make them work together.

- **layout/activity_main.xml** is UI layout for MainActivity, it define buttons, images and text on screen. I make really simple UI for my first try and later when I make my app better, I make a little bit more complicated UI.
- **values/Theme.xml** is setting theme and visual styling for the app like colors and fonts.
- **values/colors.xml** define reusable color values and text labels

I also made a folder named **assets** and this is where my model_optimized.plt file is, this is place where my app load model for classify foods.

- **AndroidManifest.xml** declares app’s settings like app name, permissions, main activity and camera access
- **build.gradle.kts (:app)** defines dependencies (like PyTorch or Kotlin), SDK versions and build settings for app.
- 
When I finally get everything work together, I get UI layout that is simple (*figure 2*)

![image](https://github.com/user-attachments/assets/42f72201-90b7-469b-b2d6-592f1ed703eb)

*Figure 2 App layout*

This layout is shown on the phone screen when the app is launched and then you can take images from your food. Under Android_app_codes is all codes I use for my app and later I will modify them when I make version 2 of this project.

Next step when my app was finally working is running **demo** and testing how well my model is making classification for pictures that is taken with phone.
