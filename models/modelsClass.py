import io
import joblib
import h5py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
   


class VGG16Model:
    def preprocess_data(self, images_path, label_file, data_file, isReduced = False):
        # Charger les fichiers CSV
        df = pd.read_csv(label_file)
        df_data = pd.read_csv(data_file)

        if isReduced:
            df = df.iloc[0:100]
            df_data = df_data.iloc[0:100]

        # Ajouter le chemin complet aux fichiers d'images
        df_data['image_path'] = df_data.apply(lambda row: os.path.join(images_path, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)

        # Séparer les images et les labels
        image_paths = df_data['image_path'].values
        labels = df['prdtypecode'].values

        # Convertir les labels en entiers
        
        labels = self.label_encoder.fit_transform(labels)

        return image_paths, labels

    def convert_to_dataset(self, paths, labels, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(self.preprocess_image).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset 

    def preprocess_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = preprocess_input(image)  # Utiliser la fonction de prétraitement de VGG16
        return image, label

    def __init__(self, input_shape=(224, 224, 3), num_classes=1000, include_top=True, weights='imagenet'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.include_top = include_top
        self.weights = weights
        self.model = self._build_model()
        self.label_encoder = LabelEncoder()

    def _build_model(self):
        base_model = VGG16(weights=self.weights, include_top=self.include_top, input_shape=self.input_shape)
        
        if not self.include_top:
            x = base_model.output
            x = Flatten()(x)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model

       # Freeze First Layers
        for layer in model.layers[:15]: 
            layer.trainable = False
        for layer in model.layers[15:]:
            layer.trainable = True

        return model

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        return self.model.summary()

    def train(self, train_data, validation_data, epochs=1):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data)

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, image_path):
        # Read the image file
        image = tf.io.read_file(image_path)
        # Decode the PNG image
        image = tf.image.decode_png(image, channels=3)
        # Resize the image
        image = tf.image.resize(image, (224, 224))
        # Preprocess the image
        image = preprocess_input(image)
        # Expand dimensions to match the input shape expected by the model
        image = tf.expand_dims(image, axis=0)
        # Predict
        #predictions = self.model.predict(image)
        return self.model.predict(image)
    
    def predict_class(self, image_path):
        predictions = self.predict(image_path)
        return self.label_encoder.inverse_transform(predictions.argmax(axis=1))
    
    def save_model(self, filename):
        """
        Sauvegarde le modèle au format .h5.
        """
        self.model.save(filename)

    def save_encoder(self, filename):
        """
        Sauvegarde l'encodeur au format .joblib.
        """
        joblib.dump(self.label_encoder, filename)

    def load_model(self, filename):
        """
        Charge le modèle depuis un fichier .h5.
        """
        loaded_model = load_model(filename)

        # Freeze almost all layers, we assume only fine tunning
        for layer in loaded_model.layers[:-2]: 
            layer.trainable = False
        for layer in loaded_model.layers[-2:]:
            layer.trainable = True

        self.model = loaded_model

        self.compile_model()

    def load_encoder(self, filename):
        """
        Charge l'encodeur depuis un fichier .joblib.
        """
        self.label_encoder = joblib.load(filename)

    @classmethod
    def from_pretrained(cls, filename):
        """
        Constructeur alternatif pour creer une instance en se basant sur un modele existant.
        """
        instance = cls()
        instance.load_model(filename)
        return instance



# Exemple d'utilisation
if __name__ == "__main__":
    fileLabels = "../data/raw/Y_train_CVw08PX.csv"
    pathImgs = "../data/raw/images/image_train/"
    dataFile = "../data/raw/X_train_update.csv"

    # Déterminer le chemin absolu du fichier actuel (main.py)
    current_dir = os.path.dirname(__file__)
    fileLabels = os.path.join(current_dir, fileLabels)
    pathImgs = os.path.join(current_dir, pathImgs)
    dataFile = os.path.join(current_dir, dataFile)

    model = VGG16Model(input_shape=(224, 224, 3), num_classes=27, include_top=False)

    img_paths, labels = model.preprocess_data(label_file=fileLabels, images_path=pathImgs, data_file=dataFile, isReduced=True)

    X_train, X_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.33, random_state=42)

    dataset_train = model.convert_to_dataset(X_train, y_train)
    dataset_val = model.convert_to_dataset(X_test, y_test)

    # Compilez le modèle
    model.compile_model()

    # Affichez le résumé du modèle
    model.summary()

    # Entraînez le modèle
    model.train(train_data=dataset_train, validation_data=dataset_val, epochs=1)

    # model.save_model("test.h5")

    model.save_model("test.h5")
    model.save_encoder("encoder.joblib")



    # Évaluez le modèle
    test_loss, test_accuracy = model.evaluate(dataset_val)
    print(f'Test accuracy: {test_accuracy}')

    model.load_model("test.h5")

    model.summary()

    model.train(train_data=dataset_train, validation_data=dataset_val, epochs=1)
    # Évaluez le modèle
    test_loss, test_accuracy = model.evaluate(dataset_val)
    print(f'Test accuracy: {test_accuracy}')

    model = VGG16Model.from_pretrained("pretrain_models/gold_vgg16.h5")
    model.load_encoder("pretrain_models/encoder.joblib")

    model.summary() 

    pred = model.predict("C:\\Users\\joan\\Documents\\DataScience_MLOPS\\Project\\mlops_rakuten2\\data\\raw\\images\\fine_tuning\\image_1006538318_product_272891494.jpg")

    print(model.label_encoder.inverse_transform(pred.argmax(axis=1)))
