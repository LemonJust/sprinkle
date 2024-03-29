# How to Run Sprinkles in Docker
The reason you want to run 'sprinkle' in Docker is probably to avoid the pain of installing Tensoflow with GPU support. Unfortunately, you still need to set up Docker with GPU support. This is a pain, but it's a one-time pain. Once you've done it, you can run any Docker image with GPU support.

Tensorfol has a nice description of how to do this [here](https://www.tensorflow.org/install/docker). 

# How to create a Docker Container for Sprinkle
## From Scratch (probably not what you want)
1. Follow instructions [here](https://www.tensorflow.org/install/docker) to install Docker with GPU support and get Tensorflow running in a container.
2. 