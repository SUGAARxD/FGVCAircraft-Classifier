# FGVCAircraft-Classifier

## Description

This project implements a deep learning classifier for image classification. The goal is to accurately classify images of aircraft into predefined categories based on their visual content.

## Technologies

The classifier is built using `PyTorch`, leveraging convolutional neural networks (CNNs) for classification.

## Architecture Details

- The model consists of **7 convolutional layers** followed by **3 fully connected layers**.

- In addition to the convolutional layers, `max pooling` layers are used, and `global average pooling` is applied between the convolutional layers and the fully connected layers. 

- **Activation Functions:** The model uses `ReLU` activations in the convolutional layers. In theory, Softmax is applied in the final layer. However, in practice, **when using Cross-Entropy loss from PyTorch, you should not apply Softmax in the final layer**. This is because Cross-Entropy utilizes an optimized version of Softmax to prevent overflows and expects raw logits as input.

- **Optimizer and Loss Function:** I use the `Adam` optimizer and `Cross-Entropy loss`.

- **Regularization:** `Dropout` layers are applied after the fully connected layers to reduce overfitting. I also use `weight decay` as a form of regularization; however, its impact appears to be neglectable in this context.

- **Data Preprocessing:** All images are `resized` to 512x512 to match the input size of the model. During training, data augmentation is applied using `random horizontal flips` to increase the dataset's variability and improve the model's generalization.

- **Evaluation Metrics:** The model's performance is evaluated using `accuracy` and `loss` on the validation set.


## Dataset

The dataset used for training is [FGVCAircraft](https://pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft.html#torchvision.datasets.FGVCAircraft). Aircraft models are organized in a three-level hierarchy:

- **Variant**
- **Family**
- **Manufacturer**

I used the second level, **family**, which consists of 70 different classes. The FGVCAircraft class from `torchvision.datasets` automatically splits the data into train, validation (val), trainval, and test sets, which can be specified via the `split` parameter. You should also specify the hierarchy level using the `annotation_level` parameter.

If you want to train the model using the other two levels from the hierarchy (variant and manufacturer), you should adjust the output neurons in the network accordingly.

## How to Use

### Installation

1. **Clone the repository**
2. **Create and Activate a Virtual Environment Using Conda:**
   ```bash
   conda create --name env_name python=3.10
   ```
   ```bash
   conda activate env_name
   ```
   Replace `env_name` with your desired environment name.
   
4. **Install the Packages and Dependencies:**

   **Using Conda:**
   ```bash
   conda install tensorboard numpy=1.26.4 tqdm pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

   **Or Using Pip:**

   ```bash
   pip install tensorboard numpy=1.26.4 tqdm torch torchvision
   ```
### Running the code

You can use a code editor like Visual Studio Code or an IDE like PyCharm. Use your environment with all packages installed.

Alternatively, you can run the script from the console:

1. **Enter the Project Folder:**
   ```bash
   cd path_to_repo\FGVCAircraft-Classifier
   ```
   **Make sure to replace `path_to_repo` with the actual path to your cloned repository.**
2. **Run the Code:**
   ```bash
   python train.py
   ```
   
   **Or**

   ```bash
   python3 train.py
   ```
   If the first command does not work.
