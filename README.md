# Feature Engineering for Enhanced Accuracy in Image Classification

This project explores the impact of feature engineering techniques to enhance digit classification accuracy using the MNIST dataset. It implements and compares Neural Networks, Linear Discriminant Analysis (LDA), and Support Vector Machines (SVM) to evaluate their effectiveness in feature extraction and classification.

## 👩‍💻 Authors

- Zaima Sohail — 2021-EE-53 — [zaimasohail20@gmail.com](mailto:zaimasohail20@gmail.com)  
- Rohiya Shafiq — 2021-EE-55 — [rohiyashafiq1020@gmail.com](mailto:rohiyashafiq1020@gmail.com)

## 🎯 Objective

To improve MNIST digit classification performance through:
- Preprocessing and reshaping image data
- Feature extraction using Neural Networks and LDA
- Classification using SVM (with linear and RBF kernels)
- Performance comparison of the different approaches

## 📁 Project Files

- `Feature Engineering for Enhanced Accuracy in Image Classification.ipynb` — Jupyter notebook containing all code and results  
- `2021-ee-53,55.pdf` — Final report of the project  
- `requirements.txt` — Python dependencies  
- `README.md` — Project overview

## 🛠️ Tools & Libraries

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## 🔍 Techniques Used

### 1. Neural Network Feature Extraction
- A simple feedforward neural network (Dense → Dropout → Output)
- Used Keras to build and train the model
- Extracted features from the last hidden layer
- Trained an SVM classifier on these features

### 2. Linear Discriminant Analysis (LDA)
- Applied LDA for dimensionality reduction
- Used both linear and RBF kernel SVM classifiers on LDA features

## 📊 Results Summary

| Method                        | Accuracy (%) |
|------------------------------|--------------|
| NN Features + SVM            | 98.08        |
| LDA Features + SVM (Linear)  | ~92.0        |
| LDA Features + SVM (RBF)     | ~91.5        |

- Accuracy and precision scores were evaluated
- Confusion matrices were plotted for better insight

## ⚠️ Challenges & Solutions

- Neural Network initially gave poor results (~13%) due to dataset shuffling
- After removing shuffling, accuracy improved to ~98%
- Difficulty understanding complex NN math led to trying LDA + SVM for better interpretability

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/zaima-20/Feature-Engineering-MNIST.git
cd Feature-Engineering-MNIST
````

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Notebook

```
jupyter notebook "Feature Engineering for Enhanced Accuracy in Image Classification.ipynb"
```

## 📦 requirements.txt

```txt
numpy
matplotlib
scikit-learn
tensorflow
```

## 📚 References

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [Kaggle SVM MNIST Example](https://www.kaggle.com/code/jnikhilsai/digit-classification-using-svm-on-mnist-dataset)

## 📬 Contact

For any queries:

* [zaimasohail20@gmail.com](mailto:zaimasohail20@gmail.com)
* [rohiyashafiq1020@gmail.com](mailto:rohiyashafiq1020@gmail.com)


