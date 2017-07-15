# DeepLearnInfilect
This repository contains examples for solving image classification problem for ANN and CNN. The code uses tensorflow library. Will come up with a keras implementation soon.

Under ANN we use a 3 layers ANN with a relu activation function and a softmax function at the output. We can reach accuracy of 97% in image classification. As of now we train on mnist images.However it is easy to train on any 28 * 28 image with this code. We can also change to accomodate for image of any size, by doing a bit matrix math and adjusting the layers accordingly. However keep in mind that if there is huge data, it is better to run on big GPU. This code has been tested on both GPU and CPU.

Using CNN we have also run the code with same data, and same modifications can be done to the code for adjusting other image size. We have used two convolutional layers with max pooling and at two fully connected layers with dropout of 0.75.
In the final layer we have a softmax function.
It is also shown how we can freeze our model and used it as a serving from tensorflow. So that we can access the trained model for predictions. It is also shown how to host it as a server. Flask is used for that.
There are other ways of serving models with tensorflow. Will be implemented soon.

A transfer learning solution from VGGNet has been added in a new folder. All the data is removed. Running the code would automatically incorporate the changes. For queries contact roydcat14@gmail.com
