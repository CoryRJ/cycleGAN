# cycleGAN
My implementation of cycleGAN (https://arxiv.org/pdf/1703.10593.pdf), included are some helper files. This uses Tensorflow.
CycleGAN is a method used to create neural networks that can translate from one domain to another. It consists of four neural networks: two generators than transform from one domain into the other and two discriminators that try to decide between real images and fake ones.


![results](https://user-images.githubusercontent.com/26369491/52381706-c47f0900-2a2f-11e9-897b-d4ac9b355aeb.png)
Using cycleGAN for Male <-> Female conversion.
The celebA dataset.
Key:
Rows 1 and 4: Original Images
Rows 2 and 5: Cycle consistency F->M->F and M->F->M
Rows 3 and 6: Domain transfer F->M and M->F

The structure of the generator I used differs slightly from that of the paper, but not significantly - I just used some extra skip layers. The current uploaded cycleGAN implementation is the most recent one I am currently working with, so to get it working on celebA you will need to update some variables.

This particular project has been shelved for something else that I am currently working on.
