# Step 6 – Demonstrating and testing

When I finally get my app work, I try it inside **Android studio** first. That first try wasn’t quite successful because the app opened but when I pressed button to take picture app crashed like we can see in video below( *video 1*)

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



