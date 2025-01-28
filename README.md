# Bell Pepper Bacterial Spot Classification

Bacterial spot is a serious disease affecting bell peppers, caused by the *Xanthomonas* species of bacteria. It leads to dark, water-soaked lesions on leaves and fruits, reducing yield and marketability. Early detection and classification of infected plants are crucial for effective disease management. 

A Convolutional Neural Network (CNN) model was developed to classify bacterial spot disease in bell peppers using images from the PlantVillage dataset.

## Dataset
The dataset used is the **Bell Pepper Dataset** from PlantVillage. It consists of labeled images of bell pepper leaves, categorized as either healthy or infected with bacterial spot disease.

## Data Splitting
The dataset is split into:
- **80% Training**
- **10% Validation**
- **10% Testing**

## Preprocessing
Before training, the dataset undergoes preprocessing to improve performance.

### Caching, Shuffling, and Prefetching
To optimize data loading, the dataset is cached, shuffled, and prefetched using TensorFlow’s `AUTOTUNE` feature.

```python
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
```

### Image Resizing and Normalization
Images are resized and rescaled to normalize pixel values.

```python
resize_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.Rescaling(1./255)
])
```

### Data Augmentation
To improve generalization, random flipping and rotation are applied.

```python
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])
```

## Convolutional Neural Network (CNN) Model
A CNN architecture is implemented using TensorFlow/Keras with the following characteristics:
- **Input Shape:** `(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)`
- **Three Convolutional Layers:**
  - Conv2D (32 filters, 3x3 kernel, ReLU activation) + MaxPooling2D
  - Conv2D (64 filters, 3x3 kernel, ReLU activation) + MaxPooling2D
  - Conv2D (128 filters, 3x3 kernel, ReLU activation) + MaxPooling2D
- **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
- **Fully Connected Layers:**
  - Dense (512 neurons, ReLU activation)
  - Dense (2 neurons, softmax activation for classification)
- **Loss Function:** Sparse categorical cross-entropy
- **Optimizer:** Adam

```python
model = models.Sequential([
    resize_rescale,
    data_aug,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
```

## Performance
The model achieved **100% accuracy** on the test dataset.

## Future Improvements
- Evaluate performance on a larger and more diverse dataset.
- Implement transfer learning using pretrained models like VGG16 or ResNet.
- Fine-tune hyperparameters to improve generalization.

## License
This project is licensed under the MIT License.

