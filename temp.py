import h5py
import keras

def get_keras_version(model_file):
    with h5py.File(model_file, 'r') as f:
        keras_version = f.attrs.get('keras_version')
        backend = f.attrs.get('backend')
        return keras_version, backend
    
if __name__ == "__main__":

    version, backend = get_keras_version("api/pretrain_models/gold_vgg16.h5")
    print(f"Keras version: {version}")
    print(f"Backend: {backend}")

    print(f"Keras version: {keras.__version__}")