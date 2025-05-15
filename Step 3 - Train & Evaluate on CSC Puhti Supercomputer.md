# Step 3 - Train & Evaluate on CSC Puhti Supercomputer

Next step was starting to train model and after training one model I made more versions because I did some **parameter tuning** to find version and model that performed best, I encountered some problems and that is why I needed to change something in models and finally I changed **PyTorch** to **TensorFlow** so finally I had 4 different versions. 

I also needed to do **Slurm files** and run code by using them inside **Puhti supercomputer**; after training the model, I did test run how well my model can classify test set images. Some versions I try first run locally in spyder at my laptop before loading them into Puhti so I can fix errors faster.

## version 1

My **version 1** is something I start my model, after version one is working, I make other versions if needed. When building AI or machine learning system, a series of numbers and parameters define the future of a system. In this assignment each line of code in this configuration serves as a crucial piece of a larger puzzle, carefully structured to train a model capable of distinguishing between **healthy** and **unhealthy** cases.

I choose that **batch_size is 32** and this is number of samples processed in one iteration during training. Batch size of 32 is quite common and balancing computational efficiency whit model stability so that’s why it’s good size to start.

**Epoch** is number of times the model will go through the entire dataset during training. **15 epochs** should provide enough iterations to refine the model’s accuracy. 15 epochs make model slow at least when I run my model first in my laptop using spyder so now after training my first model I think 15 isn’t good number to only test if model is even working because dataset is large, and it takes so much time when my laptop is not supercomputer.

**Learning_rate** controls how much the model adjusts weights in response to error. **A learning rate of 0.001 is standard**, it prevents drastic changes while ensuring steady learning. Later I tested if changing these parameters make my model more accurate.

I have only two classes, healthy and unhealthy so that’s the reason why my **num_classes is 2**. My device variable determines whether computations will run on a GPU (Cuda) or CPU based on hardware availability. GPUs make deep learning training significantly faster.

**CPU (Central Processing Unit)** is the main processor of a computer, handling general tasks like running applications, managing files and performing calculations. It’s designed for sequential processing and it’s making it great for everyday tasks but not ideal for heavy parallel computations. My model in version 1 is using CPU so that why model training is slow when epochs are 15.

![image](https://github.com/user-attachments/assets/be7dd3e4-9f2d-4773-a232-a655685f6052)

*Figure 1 How my model training starts*

**GPU (Graphics Processing Unit)** specializes in handling multiple calculations at once. Originally designed for rendering graphics, GPUs are now widely used in machine learning, gaming and scientific simulations because they can process data in parallel, speeding up tasks like deep learning training. 

***Code for version 1 is named:Puhti_CNN_model_V1.py***

## Version 2

Version 1 CNN training pipeline is already solid and follows best practices for fine-tuning a pretrained model (**ResNet18**). There is still some ways to make it even better and improve model performance and training quality. There are some steps to try to make models better and they are listed below.

•	**Data Augmentation:** This I have done already when I split my training set, validation set and test set so this part I can skip in my version 2.

•	**Learning Rate Schedule:** Use a learning rate scheduler to reduce the LR when performance plateaus. A **learning rate scheduler** is a technique used in training neural networks to **adjust the learning rate dynamically** based on model performance. This helps prevent overshooting optima and ensures the model continues improving even when progress slows.

•	**Track F1-score and Confusion Matrix:** Accuracy can be misleading with imbalanced datasets. Confusion Matrix and F1-Score give more information on how model works and not only looking accuracy.

•	**Model Improvements:** ResNet18 is good, but you could try deeper or more specialized models if you have GPU capacity. For example,
    o	Options:
        - ResNet34, ResNet50 (deeper versions)
        - efficientnet_b0 to b3 (more efficient)
        - vit_b_16 (Vision Transformer, if using Torch 2.0+ and good GPU)
        
•	**Mixed Precision Training (If on GPU):** To speed up training and reduce memory usage

•	**Model checkpointing:** Save optimizer state and epoch for resuming training. **Model checkpointing** is a technique used to **save training progress**, allowing you to resume training later without starting from scratch. This is especially useful when training deep learning models, which can take a long time to converge.

•	**Visualize Predictions:** Add code to visualize correct/incorrect predictions with TensorBoard, this is useful to see models performing in visual way and it might give some information that numerical output is not showing clear way.

In **version 2** I try to make my model more accurate and perform better, so I add some features that are listened to below. Because the time frame is short, I make only three version where I try to make my model perform better. This cause that I didn’t have so many chance to fine tune my hyperparameters.

**Features Added:**

1. Learning Rate Scheduler (ReduceLROnPlateau)
2. Evaluation Metrics: classification_report & confusion_matrix
3. Improved Model: Using resnet50 instead of resnet18
4. Mixed Precision Training (with torch.cuda.amp)
5. Model Checkpointing: Includes optimizer and epoch
6. TensorBoard Visualization of Wrong Predictions

When my version 2 was running I noticed that this version is not as good as version 1 so I’m using version 1 in my test 1.

***The code for version 2 is named: Puhti_CNN_model_V2.py***

## Version 3

I also made version 3, and I tried out that I added more accuracy and make model even better.

**Architectural improvements** I did for version 3 are: **Classifier head** I change into a **linear + dropout (0.5)** that it reduce overfitting, also I did **layer freezing** that is doing **Frozen backbone + trained last layer**, this is done for better transfer learning. 

Some **data pipeline upgrades** that I did was more augmentations into images, in **version 2** I had only basic **flip/rotation/resize** and in **version 3** I have **advanced augmentation like color jitter and random crops**. I also add **class balancing** with *weightedRandomSampler* and *normalization* by using **ImageNet stats**.

**Training Optimization** I change my loss function from plain **crossEntropy to Weighted + Label Smoothing (0.1)**, I also change **optimizer** from Adam to **AdamW( weight decay = 1e-4)**. I improve **LR Scheduler to ConsineAnnealingLR** and batch size in version 3 is 64. In version 3 I also have Gradiet handling that is done by **Clipping (max_norm = 1.0)**.

**Expected performance gains in version 3:**

1. **Training Stability:** Fewer divergences (gradient clipping + better LR scheduling)
2. **Generalization:** Better test performance (advanced augmentations + label smoothing)
3. **Class Balance:** Improved minority class recall (weighted sampling + loss)
   
When I run my version 3 Slurm file it end into epoch 6 and the result was not good, so I ended test in that, later when I have more time, I do more tests and update my model then.

***Code for version 3 is named: CNN_model_V3.py***

## Version 4 - TensorFlow model

I get problems when I start building my mobile optimization code and android app, When I try do many things at the same time, I notice that I have a problem. My model is using **PyTorch**, and my **android studio codes are for Tensorflow lite**. So, I decided to do code that is using **Tensorflow instead of PyTorch**, I decided this because I haven’t used Android studio ever and when I finally get it work without errors, I don’t  want to start messing it up by changing everything into PyTorch style.

For optimizing I need new code also, so I just use AI to modify my old codes that are for PyTorch to work with TensorFlow.

This model uses **ResNet50, batch size 32 and number of epoch are 15, learning rate is 0.001**. So, the model is quite the same as PyTorch model when we look at hyperparameters, but TensorFlow does not support ResNet18, so it must be ResNet50.

***The code for version 4 is named: Puhti_CNN_model_tensorflow_V1.py***

## Version runs in Puhti

**Puhti  support GPU computing**. It uses **NVIDIA Volta V100 GPUs**, with 80 GPU nodes, each containing 4 GPUs, totaling 320 GPUs. **Puhti is optimized for GPU-accelerated machine learning, supporting frameworks like TensorFlow and PyTorch**. When I run my code in Puhti I need to submit jobs using the Slurm workload manager and specify GPU resources.

There is two way to get Puhti working with code, **Slurm file style** and then **jupyter notebook style** that needs its own settings.

### SLurm file style

**Slurm scripts** are batch job submission files used in **high-performance computing (HPC) environments like Puhti**. They're written in **Bash** and contain instructions for scheduling and executing computational tasks. A **Slurm file** (*often .sh format*) tells the **Slurm workload manager**: 

- What **resources** (CPU, GPU, memory, time) do your job needs
- Where to **save logs** (output/error files)
- What **commands** to execute (e.g., running your Python script)
      
Slurm files **automate job submission** to computing clusters, allowing **Parallel processing** Run tasks across multiple CPUs/GPUs

In Slurm style I created a file that is named *train_puhti_V1.sh*, in this file I have information and code for training a CNN model on the **CSC Puhti supercomputer**.

![image](https://github.com/user-attachments/assets/1d06659f-26e8-41a3-bfb2-d72746f1f6ff)

*Figure 2 Slurm file*

In file I have lines and below I explain all those lines:

1. define the script as a Bash script
2. The name of the job is visible in the queue system.
3. CSC project ID — all job usage is billed to this project.
4. Requests a GPU node.
5. Requests 1 NVIDIA V100 GPU.
6. Allocates 4 hours of runtime.
7. Reserves 32 GB of RAM.
8. Reserves 4 CPU cores (useful for data loading).
9. Saves stdout/stderr logs to output.log.
10. .
11. .
12. Loads PyTorch version 2.6 environment on Puhti.
13. .
14. .
15. .
16. Runs your training script located at: /projappl/project_XXXX/USER_NAME/codes/Puhti_CNN_model_V1.py

And so Slurm file can load Puhti_CNN_model_V1.py it needs to be load into Puhti and that happen in **WSL ubuntu prompt** using command: 

*scp "/mnt/c/file path" username@puhti.csc.fi:/projappl/project_XXXXX/file_name*

After running this file in Puhti prompt by using command *sbatch train_puhti.sh* I have file **Output.log** that is telling me result of code (*figure 3*)

![image](https://github.com/user-attachments/assets/54d2b04f-9ea5-4e14-b4eb-a63d23332da8)

*Figure 3 Output.log file*

I also got the *best_model.pt* and *final_model.pt* that I use then later when I optimize my model for mobile phone.

## Running sets code and test set

After the three version I decided to use **version 1 best model** because it worked best. So, for running test set I use *best_model.pt* from version 1.

For testing how my model work I run code first locally after I have downloaded my *best_model.pt* from puhti, I use my OneDrive file that have my test images and those are same images I have in my puhti data.

After running **test 1** I get the result that is shown in figure below, this shows that my model accuracy is only **85.73%** and it is ok for now, I use this for my mobile optimization and my android app.

![image](https://github.com/user-attachments/assets/0b3dbad7-714f-4cf1-a336-cf88aa906b55)

*Figure 4 Test set accuracy*

Time frame is so short that I can’t run more tests and fine tune my model more even if I want, code running take so long that I don’t have enough time run it many times.

***The code for test set run is named: Puhti_CNN_model_test1.py***

## Running test set with TensorFlow

After getting my *model.h5* file I run test code for testing how accurate my model is when I use it in my test set.

***The code for test set run is named: test_tflite_model.py***
