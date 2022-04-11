import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('./ModelData/')

print(model.summary())
sample0 = {
    "size": 18.9,
    "flag1": 1,
    "flag2": 1,
    "flag3": 0,
    "flag4": 0,
}

sample1 = {
    "size": 3.8,
    "flag1": 0,
    "flag2": 0,
    "flag3": 1,
    "flag4": 0,
}

input_dict1 = {name: tf.convert_to_tensor([value]) for name, value in sample0.items()}
input_dict2 = {name: tf.convert_to_tensor([value]) for name, value in sample1.items()}

predictions1 = model.predict(input_dict1)
predictions2 = model.predict(input_dict2)

print("prediction1", predictions1)
print("prediction2", predictions2)