---
layout: page
---

## Diffusion Models I
---
Diffusion models are another approach to generative modelling. The algorithm became popular with the release of [Dalle-2](https://arxiv.org/abs/2204.06125) and [Stable Diffusion](https://arxiv.org/abs/2112.10752). However, as we will see the idea is been around some time already.


In this notes I will not focus on the details of the current SOTA algorithms but in the mathematical foundations of the idea. I will not describe _conditional_ (image) generation. I will focus on approximating the probability distribution of a given dataset.

As an exercise I translated the 'pytorch' code to 'tensorflow', hence all code here is in 'tensorflow'.

These notes are (mostly) based on:

[1] This excellent repository: https://github.com/acids-ircam/diffusion_models

[2] This awesome article: https://arxiv.org/abs/2208.11970

[3] The work from Yang Song, who accompanies his research with super helpful blogposts: https://yang-song.net/blog/2019/ssm/, https://yang-song.net/blog/2021/score/ with the paper https://arxiv.org/abs/1907.05600.

[4] Some classic papers on the topic: https://arxiv.org/pdf/2006.11239.pdf , https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf, https://arxiv.org/abs/1505.04597

[5] Some other blogposts: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ 

### Introduction, some intuition
---
The name diffusion already gives a clue about the underlying idea: Reversing a _diffusion_ process. In order to do so we parameterize a function that is able to go from pure noise (end point of the diffusion process) to the original coherently structured substance. In the sense of going from noise to coherent data diffusion models are similar to GAN, but the similarities end there. 

![Alt Text](/assets/images/DiffusionModels/DiffusionDalle2.png)

Another way of looking at this, motivated by the _score modelling_ point of view, is the idea of "surfing" a high dimensional space towards the areas where the coherence (w.r.t our data) within the dimensions is maximized. Or, what is the same, climbing a high dimensional probability distribution towards the peak areas. To clarify, high probability regions (the peaks) in this space are where the combination of the dimensions is more likely to render an observation belonging to our dataset. I think this is quite an interesting though!

The next image (which is constructed by the code showed here) describes this process of surfing the 28x28 dimensional space of the MNIST dataset towards high probability regions... consequently reconstructing a coherent image!

![Alt Text](/assets/images/DiffusionModels/MNIST_SAMPLING.gif)

But, how on earth we can do this? 

TODO: 
-- Talk about the idea of the gradient...
-- Some final comments for introducing the rest. From score-modelling to diffusion stuff

## Score modelling
---
