EM algorithm for GMM (gaussian mixture model)
=============================================

Tensorflow implementation of [EM-algorithm for GMM](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model).

<div class="text-center">
  <img src="./assets/Figure9.8a.png" alt="Figure 9.8a" height="180px" />
  <img src="./assets/Figure9.8b.png" alt="Figure 9.8b" height="180px" />
  <img src="./assets/Figure9.8c.png" alt="Figure 9.8c" height="180px" />
</div>

<div class="text-center">
  <img src="./assets/Figure9.8d.png" alt="Figure 9.8d" height="180px" />
  <img src="./assets/Figure9.8e.png" alt="Figure 9.8e" height="180px" />
  <img src="./assets/Figure9.8f.png" alt="Figure 9.8f" height="180px" />
</div>

This implementation contains both EM-algorithm and Gradient descent algorithm.

Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [TensorFlow](https://www.tensorflow.org/)


Usage
-----

To use EM-algorithm

    $ python main.py 

To use gradient descent

    $ python main.py --use_GD True
    
Results
-------
![result1](./assets/result1.png)
![result2](./assets/result2.png)
![result3](./assets/result3.png)
![result4](./assets/result4.png)
![board1](./assets/board1.png)
![board2](./assets/board2.png)

Author
------

Kyowoon Lee / [@leekwoon](http://leekwoon.github.io/)