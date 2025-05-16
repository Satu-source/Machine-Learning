# Step 4 –  Optimize for Mobile Device

**Optimizing my trained model for mobile deployment** — is a crucial part of  project. Since I’m using a CNN (like ResNet18), the standard way to optimize it for Android or iOS deployment is:

1. Quantization (critical for mobile)
2. Mobile Export format
   
    a.	I use PyTorch
   
4. Platform – Specific Deployment
5. Performance benchmarks
   
    a.	Latency
   
    b.	Accuracy drop
   
    c.	model size

I download my **best_model.pt** from my puhti and decided to run locally my optimization code, it take less time to do it that way then wait Puhti server to run my code.

When I run my optimization code, I need to make file **model_optimized.ptl** so I can then load that file to **Android Studio** for my app, I also want make benchmarking how my **accuracy drop** when I optimize my model and see **how big my model** is and what is my **average latency**. In *figure 1* is my result when I optimize my model by using PyTorch.

![image](https://github.com/user-attachments/assets/37977301-8bde-403d-9b1f-8ee14e08d0e9)

*Figure 1 Mobile optimization result*

Result is good looking, the accuracy drop is only **0.08%** and that is acceptable for mobile deployment after quantization and optimization. Model size is **44.68MB**, this is a decent size for mobile interference, but it could be still reduced if needed. Latency is **123.97 ms** and this is good for a CPU- only setup, it is also acceptable for many real-time or near real-time applications and my camera application is real-time application.

After running this one optimization, I started applying it to my android studio and app to get working results. 

***The code for this optimization is named: PyTorch_mobile_optim_v3.py***
