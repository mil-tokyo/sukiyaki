# Sukiyaki JavaScript Library
The Fastest Deep Learning Library for JavaScript

Sukiyaki is being developed to be the fastest Deep Learning Library for JavaScript. It uses Sushi ( https://github.com/mil-tokyo/sushi ) inside for the fast matrix calculation.

## Technical Features

### Support of GPGPU and multi core CPU
You can use GPGPU and multi core CPU via Sushi. If your system does not support WebCL, Sukiyaki uses standard JavaScript automatically.

### Support of Deep Convolutional Neural Network

## How to Try

### initialize and download dataset

  git submodule init
  git submodule update
  ./sample/dataset/mnist/download.sh
  # or
  ./sample/dataset/cifar/download.sh

### try sample program with node.js

  node ./sample/node/main
  > mnist

### try sample program with browsers

  ./sample/browser/server.sh
  and access http://localhost:64649/sample/browser

If you would like to use GPGPU, check Sushi ( https://github.com/mil-tokyo/sushi ) and install a WebCL implementation.
