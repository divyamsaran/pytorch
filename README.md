# pytorch
Hey! Welcome to my repository on Pytorch, I have tried implementing CNNs, RNNs, GANs, and transfer learning using Pytorch. Disclaimer: a lot of the content is inspired by the amazing [Udacity course on Pytorch](https://classroom.udacity.com/courses/ud188).

You will find the following on the repository:
1. Convolutional Neural Networks (CNN): I have created a simple CNN for CIFAR-10 dataset that achieves 74% accuracy on the test set. The network consists of three convolution layers and two fully connected layers. RELU is the activation function of choice, and I have used Adam optimizer and Cross Entropy Loss. 
2. Recurrent Neural Networks (RNN): I have created a character level LSTM, which is a Pytorch implementation of Andrej Karpathy's popular [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on LSTMs.
3. Generative Adversarial Networks (GANs): I have created a simple GAN to generate numbers similar to MNIST data. Both the Generator and Discriminator are 4 fully connected networks. 
![GAN](https://github.com/divyamsaran/pytorch/tree/master/gan/results/gan_pytorch.gif)
4. Transfer Learning: I use the pretrained VGG-16 model from Pytorch and change it's classifier layer to learn on Flower dataset. With transfer learning, the model achieves 78% accuracy in 5 epochs.