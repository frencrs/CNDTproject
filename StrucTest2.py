import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('./ModelData/')

print(model.summary())
sample0 = {
    "avg_len": 604.78,
    "num_unique_ports": 46,
    "num_unique_dst": 18,
    "num_unique_src": 18,
    "num_unique_ttl": 8,
    "num_unique_chksum": 3160,
    "packet_count": 3467,
}


input_dict1 = {name: tf.convert_to_tensor([value]) for name, value in sample0.items()}
#input_dict2 = {name: tf.convert_to_tensor([value]) for name, value in sample1.items()}

predictions1 = model.predict(input_dict1)
#predictions2 = model.predict(input_dict2)

print("prediction1", predictions1)
#print("prediction2", predictions2)