import tensorflow as tf

# Dataset path
DATASET_PATH = "prepared_dataset"

# Image size required by MobileNetV2
IMAGE_SIZE = (224, 224)

# Number of images processed together
BATCH_SIZE = 32

# Training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + "/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

# Validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + "/validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nDataset Loaded Successfully!")

print(f"\nTraining batches   : {len(train_dataset)}")
print(f"Validation batches : {len(validation_dataset)}")