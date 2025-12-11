# 😠😄 Emotion Recognition with CNN from Scratch  
A full **Convolutional Neural Network** trained **from zero** (no transfer learning) to classify **7 human emotions** using the FER-style dataset. The model handles **heavy class imbalance**, applies **strong augmentations**, and includes **custom training loops**, evaluation, and TorchScript export.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Exporting the Model](#exporting-the-model)
- [Results](#results)
- [License](#license)

---

## 🚀 Overview
This project trains a custom CNN to classify the following **seven emotions**:

| Emotion | Label |
|--------|-------|
| Angry 😠 | 0 |
| Disgust 🤢 | 1 |
| Fear 😱 | 2 |
| Happy 😄 | 3 |
| Neutral 😐 | 4 |
| Sad 😢 | 5 |
| Surprise 😲 | 6 |

Unlike typical emotion-recognition repos, this one **does not rely on pretrained models** like ResNet or VGG.  
You built everything from scratch — which means you actually learned something instead of copying Keras tutorials.

---

## 📂 Dataset  
The dataset structure (after extraction):

```
train/
 ├── angry/
 ├── disgust/
 ├── fear/
 ├── happy/
 ├── neutral/
 ├── sad/
 └── surprise/

test/
 ├── angry/
 ├── disgust/
 ├── fear/
 ├── happy/
 ├── neutral/
 ├── sad/
 └── surprise/
```

Downloaded via:

```bash
gdown 1oTQE8pGkq9rEvCLs89lYUjoIpjPgTzke
unzip archive.zip
```

---

## ⭐ Key Features

- ✔ **Custom CNN** (no pretrained networks)  
- ✔ **Class imbalance solved** using computed class weights  
- ✔ **Data augmentations** (ColorJitter, rotations, flips)  
- ✔ **Weighted CrossEntropyLoss**  
- ✔ **Custom training & testing loops**  
- ✔ **TorchScript export (`model_emotion.pt`)**  
- ✔ **Full evaluation**: accuracy, classification report, confusion matrix  

---

## 🔧 Installation

```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm scikit-learn gdown
```

---

## 📁 Project Structure

```
emotion-recognition/
│── train/
│── test/
│── Emotion_CNN.ipynb
│── model_emotion.pt
│── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

A stacked CNN built with:
- BatchNorm  
- Conv → ReLU blocks  
- MaxPooling  
- Flatten → Linear (7 classes)  

```
[Conv → ReLU] x 2  
↓ MaxPool  
[Conv → ReLU] x 2  
↓ MaxPool  
[Conv → ReLU] x 2  
↓ MaxPool  
↓ Flatten  
↓ Linear(256*2*2 → 7)
```

This is not a toy network. It’s deep enough to learn real emotional features and fast enough to train on CPU if necessary.

---

## 🏋️ Training

Run training:

```python
train_loss, train_acc, test_loss, test_acc = train_model(
    model,
    DataLoaderTrain,
    DataLoaderTest,
    optimizer,
    loss_function,
    device,
    epochs=60
)
```

Loss function with class weights:

```python
loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
```

---

## 📊 Evaluation

Generate predictions & metrics:

```python
y_true, y_pred = eval_model(loaded_model, DataLoaderTest, device)
print(classification_report(y_true, y_pred))
```

Confusion matrix:

```python
ConfusionMatrixDisplay(c, display_labels=map_emotion.keys()).plot()
```

---

## 📦 Exporting the Model

TorchScript export:

```python
script = torch.jit.script(model)
script.save("model_emotion.pt")
```

Load it anywhere:

```python
loaded_model = torch.jit.load("model_emotion.pt")
```

---

## 📈 Results

Expect:
- Consistent improvement during training  
- Solid accuracy on dominant classes  
- Some difficulty with minority classes (your weighting helps but doesn’t magically solve imbalance)  

Example prediction:

```python
loaded_model(image.unsqueeze(0).to(device))
```

---

## 📜 License
MIT License – use it, modify it, or build on top of it.

---

If you also want:
✅ A **cleaner architecture diagram**,  
✅ A **GIF demo**,  
✅ Or a **more polished GitHub badge header**,  
just tell me — you should present this project like you actually want employers to notice it.
