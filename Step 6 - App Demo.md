# Step 6 – Demonstrating and testing

When I finally get my app work, I try it inside **Android studio** first. That first try wasn’t quite successful because the app opened but when I pressed button to take picture app crashed like we can see in video below( *video 1*)

***(Videos open in YouTube when images are clicked)***

[![image](https://github.com/user-attachments/assets/a64bf34f-add2-4f74-a29d-9b898d022b89)](https://youtube.com/shorts/rvuWrAJOjBo)

*Video 1 First try to use app inside Android Studio*

After this try I know that at least my app opens and it’s working somehow so next I try fix my codes that app is not crashing when I try press button to take picture. 

Next version was working better, I made some changes in my codes inside files that I explain in **step 5**. Now my app open and it’s not crashing anymore when I press *“take image”* button, app ask permission to use camera and then its opening camera and taking picture. So finally, I have an app that is working and not crashing, after all fighting with the codes and Android studio this was small winner for me.

[![image](https://github.com/user-attachments/assets/79b004d1-80cf-41bf-8cec-1ec2e5429068)](https://youtube.com/shorts/AnbDKSj5RGw)

*Video 2 Second version of app inside Android Studio*

So next I need to try an app in my own phone, I change my phone settings so I can use **developer settings** and after that I install my app using USB cable to my phone. When I have an app on my phone, I start testing how well my model works. The app look in my phone same as it looks in **step 4 layout image** so code for that works well. *Figure 1* shows how the app look on my phone.

![image](https://github.com/user-attachments/assets/d1b1f5a3-4841-421d-86d4-859fdde43ded)

*Figure 1 How app look on my phone*

I took pictures of eggs, sandwich cake and grapes. I see that my model give **accuracy between 3.0 % to 3.2%** and it’s not good at all, I don’t know what happened to my model while I did app. When I optimize my model, it was  **85.64%** but now suddenly when I take images, models can’t classify them really. First, I think it might be because of the food that I take pictures of. Below in *figure 2*, *figure 3* and *figure 4* is my apps result from those foods.

![image](https://github.com/user-attachments/assets/b4e2fd66-5a7e-4ae8-8469-823cd36e3217) *Figure 2 Grape’s result* ![image](https://github.com/user-attachments/assets/27170fdf-de01-4dfa-98c6-f135e2f2544c) *Figure 3 Egg result* ![image](https://github.com/user-attachments/assets/202f6917-9008-4c19-8c3b-e1e77c6235f6) *Figure 4 Sandwich cake result*

Below is video that I take with my phone when I try my app so in video it is seen real-time predictions that the app is doing.

[![image](https://github.com/user-attachments/assets/22ffaa36-9b23-4204-ba8f-e3834f5e20a4)](https://youtube.com/shorts/nY8F3TV66UI)

*Video 3 App in my phone and real-time predictions*

So, to make sure that my food is not a problem I decided to take images from the internet to make sure that they are something I really use in my model, I take apple, cake and hamburger images and then try my app on those. The result is the same, so my app does not use my model well. In *figure 5*, *figure 6* and *figure 7* shows that accuracy was the same **between 3.0% to 3.2%** even if I use foods that I know that I did have in my dataset.




![image](https://github.com/user-attachments/assets/a44116fc-bb26-4991-9c8f-6f28feb66a91) *Figure 5 Apple result* ![image](https://github.com/user-attachments/assets/602893f5-36ca-4e0e-9927-d31aee5b2407) *Figure 6 Cake result* ![image](https://github.com/user-attachments/assets/34a40be8-f34f-49a2-a392-458643fc6e64) *Figure 7 Hamburger result*

I didn’t have time to look more at why my model is not working in this assignment and also, I didn’t try to make my model even smaller like instructions say that I should try make **30 MB**, I just keep my model at **44.68 MB**. But making this assignment was fun and I challenge myself a lot so in future some point when my studies are at that point I want to try to make this model and app better.

In this assignment I use the first time **CSC Puhti supercomputer** and **Android Studio**, I also use **GitHub** first time at this scale that I made documentation there so this assignment teach me lots of things not just machine learning but other skills also.

- **Puhti**, I learned how to use Slurm files to run codes, and I also need to remind myself again how to use WSL and ubuntu to load files inside Puhti project.
- **Android Studio** teach me how to build apps and how important it is to understand the importance of different files and how to use my app demo in my phone.
- **GitHub**, I learn how to do documentation and how to build clear and well-structured documentation, I also learn how to use **Git GUI** to bash large datasets inside GitHub.

I’m not totally happy with my final result because the app is not doing classification well, but I can always later do version that is working better and at least I get lots of experience and in 9 days I finish this task even though it’s not perfect.
