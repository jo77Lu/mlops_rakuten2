import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

class VGG16Model:
    def __init__(self, num_classes):
        """
        Initialise le modèle VGG16 avec un nombre de classes spécifié.
        """
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Construit le modèle VGG16 avec une couche de sortie personnalisée.
        """
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze les couches du modèle de base
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, validation_data, epochs=10, batch_size=32):
        """
        Entraîne le modèle avec les données spécifiées.
        """
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_data,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_generator = val_datagen.flow_from_directory(
            validation_data,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping]
        )
        
        return history

    def evaluate(self, test_data, batch_size=32):
        """
        Évalue le modèle sur les données de test.
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        score = self.model.evaluate(test_generator)
        return score

    def predict(self, image):
        """
        Effectue une prédiction sur une image unique.
        """
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
        image = image / 255.0  # Normaliser
        predictions = self.model.predict(image)
        return predictions

    def save_model(self, filename):
        """
        Sauvegarde le modèle au format .pckl.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        """
        Charge le modèle depuis un fichier .pckl.
        """
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        # Recompiler le modèle après le chargement
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Exemple d'utilisation
if __name__ == "__main__":
    num_classes = 10  # Nombre de classes pour la classification
    model = VGG16Model(num_classes)

    # Entraînement et évaluation
    model.train('path_to_train_data', 'path_to_validation_data', epochs=5)
    score = model.evaluate('path_to_test_data')
    print(f"Score: {score}")

    # Sauvegarder le modèle
    model.save_model('vgg16_model.pckl')

    # Charger le modèle
    model.load_model('vgg16_model.pckl')

    # Faire des prédictions
    image = ...  # Chargez ou pré-traitez une image ici
    predictions = model.predict(image)
    print(f"Prédictions: {predictions}")
