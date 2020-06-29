# flowers-tpu

Train and infer models using Google v3-8 TPUs.

## About TPUs
* TPUs are powerful hardware accelerators specialized in deep learning tasks. They were developed (and first used) by Google to process large image databases, such as extracting all the text from Street View. The latest Tensorflow release (TF 2.1) was focused on TPUs and they’re now supported both through the Keras high-level API and at a lower level, in models using a custom training loop. 
* The TPUv3-8 is actually 4 chips connected together where each chip is physically similar to the GPU V100 ($8,500), i.e. they both have 32GB memory and similar compute. If you connect 4xGPU V100, then you have a similar comparison to the TPUv3-8 and you will observe that they both operate at similar speeds and similar batch size capacity.
* The TPU will perform matrix multiplications (i.e. dense and convolutional layers) on the hadrware matrix multiplication unit (MXU). The MXU works with bfloat16 inputs, float32 accumulators and float32 outputs. Conversion of your matrices from float32 to bfloat16 is performed by the MXU automatically. Then multiplying two bfloat16 numbers together naturally produces a float32 result which is kept as float32. All additions happen in float32 and the final result is float32.

## Dataset
* We're classifying 104 types of flowers based on their images drawn from five different public datasets. Some classes are very narrow, containing only a particular sub-type of flower (e.g. pink primroses) while other classes contain many sub-types (e.g. wild roses).
* Dataset link - https://www.kaggle.com/c/flower-classification-with-tpus/data

## Models
* Pretrained EfficientNets are taken as backbone
* EfficientNetB7 and EfficientNetB6 are ensembled together to give better scores.
