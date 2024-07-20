import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vit

TRAIN_PATH = 'data/train'
VALID_PATH = 'data/valid'

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary' 
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary' 
)

def create_vit_model(input_shape=(224, 224, 3)):
    base_model = vit.ViT(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

model = create_vit_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

model.save('vit_model.h5')
print("Training completed and model saved.")
