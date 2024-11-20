
# Project Dynapse

Traditional computer vision models for image recognition tasks use deep learning architectures like convolutional neural nets or transformers. Spiking neural networks (SNNs), offer a more biologically-plausible approach, as well as computational efficiency and portability compared to traditional architectures. The goal of this project, is to build a SNN using the SNNTorch library to represent visual objects from datasets like Fashion-MNIST and ORL. The model structure is based on the original implementation by Melani Sanchez-Garcia, Tushar Chauhan, Benoit R.
Cottereau, and Dr. Michael Beyeler in the paper *"Efficient Visual Object Representation Using A Biologically Plausible Spike-Latency Code and Winner-Take-All Inhibition"*, which is linked below.  


### Model

- **Architecture** : Spiking Neural Network
- **Encoding Mechanism** : Spike-Latency Coding (Temporal)
- **Training Dataset** : Fashion-MNIST (28x28, 1000 training images)
- **Preprocessing** : Center-Surround Receptive Fields represented by DoGs
- **Activation Scheme** : WTA - I 
- **Training Methodology** : Spike Timing Dependent Plasticity
### Papers

[Efficient Visual Object Representation Using A Biologically Plausible Spike-Latency Code and Winner-Take-All Inhibition](https://arxiv.org/pdf/2212.00081)

[Spiking Neural Networks: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9787485&tag=1)

### Relevant Articles 

[Implementation of Relevant Receptive Fields Using Difference of Gaussian Kernels](https://medium.com/@lsampath210/implementation-of-retinal-receptive-fields-using-difference-of-gaussian-kernel-6e13778b3ec)

