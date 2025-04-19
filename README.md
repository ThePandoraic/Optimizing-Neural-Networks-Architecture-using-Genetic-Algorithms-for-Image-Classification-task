# OPTIMIZING NEURAL NETWORKS ARCHITECTURE USING GENETIC ALGORITHMS FOR IMAGE CLASSIFICATION TASK

## OVERALL

In this project, Genetic Algorithm (GA) is used to optimize the performance of an Artificial Neural Networks (ANNs). 
The implementation is based on PyTorch in order to utilizes its GPU computing.

The optimization focus on choosing appropriate neural networks architecture as well as learning rates and activation fucntions.

The dataset used in this project is [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The dataset contains more than 7000 images in both (Training and Testing set) with 4 labels in total. 

## Data preparation and preprocessing
The original dataset is downloaded and stored in the directory `dataset/Original`. After preprocessing, the new dataset is stored at `dataset/AfterPreprocess`. 

**Dataset structure**
```
dataset
|-- Original
|        |-- Training
|        |       |-- glioma     (Tr_gl_xxxx.jpg)    --> 1321 images
|        |       |-- meningioma (Tr_me_xxxx.jpg)    --> 1339 images
|        |       |-- notumor    (Tr_no_xxxx.jpg)    --> 1595 images
|        |       |-- pituitary  (Tr_pi_xxxx.jpg)    --> 1457 images
|        |-- Testing
|        |       |-- glioma     (Te_gl_xxxx.jpg)    --> 300 images
|        |       |-- meningioma (Te_me_xxxx.jpg)    --> 306 images
|        |       |-- notumor    (Te_no_xxxx.jpg)    --> 405 images
|        |       |-- pituitary  (Te_pi_xxxx.jpg)    --> 300 images
|-- AfterPreprocess
|        |-- Training
|        |       |-- glioma     (Tr_gl_xxxx.jpg)    --> 1321 images
|        |       |-- meningioma (Tr_me_xxxx.jpg)    --> 1339 images
|        |       |-- notumor    (Tr_no_xxxx.jpg)    --> 1595 images
|        |       |-- pituitary  (Tr_pi_xxxx.jpg)    --> 1457 images
|        |-- Testing
|        |       |-- glioma     (Te_gl_xxxx.jpg)    --> 300 images
|        |       |-- meningioma (Te_me_xxxx.jpg)    --> 306 images
|        |       |-- notumor    (Te_no_xxxx.jpg)    --> 405 images
|        |       |-- pituitary  (Te_pi_xxxx.jpg)    --> 300 images
|        |-- augmented_img_paths.json
```

The preprocessing phase is handled by class **ImgPreprocess**, including:
- Crop image to center the brain.
- Augment the image with the ratio 0.25 (meaning that 25% image of each class will be augmented) and store the path of the augmented images in `dataset/AfterPreprocess/augmented_img_paths.json`
- Crop the image to the size (255, 255)

## Dataset and Dataloader
The dataset is handled by class **BrainTumorDataset** which utilizes **torch.utils.data.Dataset**. The dataloader is handled by class **Loader** which utilizes **torch.utils.data.Dataset**. The batch size is set to 32 and 20% of the training set is kept for validation.

## Dynamic Neural Networks
The general architectures of the neural networks used in this use cases is that there are several Convolutional Neural Networks (CNNs) are stacked  together, followed by some Fully Connected Layers to produce the output. 

This architecture is handled by the class **DynamicNN**. With this implementation, it allows the architecture of the networks to grow or shrink freely as well as the learning rate or its activation fucntion, depending on the optimization of the genetic algorithm.

## Genetic Algorithm
The genetic algorithm plays as an optimizer to optimize the performance of the DynamicNN. This optimization is based on randomly growing or shrinking the architecture of the DynamicNN, changing the activation function or finding another suitable value for learning rate. All of these things are handled by class **GAOptimizer**.

## Performance Metrics
- Fitness score: 0.9483
- Final Test Accuracy: 98.60%

With a fitness score of 0.9483 and a final test accuracy of 98.60%, the ANN, guided by GA, can effectively learn meaningful feature representations and achieve superior classification performance. The training loss values, which progressively decrease across epochs, suggest stable and efficient learning

## Conclusion
The Genetic Algorithm (GA)-optimized Artificial Neural Network (ANN) demonstrates strong effectiveness in brain tumor MRI classification by dynamically evolving an optimal network topology. By systematically optimizing key parameters such as layer depth, neuron distribution, and activation functions, the GA enables the ANN to efficiently learn complex spatial patterns inherent in MRI images. The model’s high test accuracy and stable learning process indicate its robustness, making it well-suited for high-stakes medical applications where precision is critical.