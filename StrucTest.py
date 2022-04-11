import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.layers import IntegerLookup
from keras.layers import Normalization
from keras.layers import StringLookup


input_dataframe = pd.read_csv("TestData2.csv")

val_dataframe = input_dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = input_dataframe.drop(val_dataframe.index)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

print(train_ds.__len__())

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


# Numerical features encoded to be encoded as floating point
size = keras.Input(shape=(1,), name="size")
# Categorical features encoded as integers
flag1 = keras.Input(shape=(1,), name="flag1")
flag2 = keras.Input(shape=(1,), name="flag2")
flag3 = keras.Input(shape=(1,), name="flag3")
flag4 = keras.Input(shape=(1,), name="flag4")


all_inputs = [size, flag1, flag2, flag3, flag4]

size_encoded = encode_numerical_feature(size, "size", train_ds)

flag1_encoded = encode_categorical_feature(flag1, "flag1", train_ds, False)
flag2_encoded = encode_categorical_feature(flag2, "flag2", train_ds, False)
flag3_encoded = encode_categorical_feature(flag3, "flag3", train_ds, False)
flag4_encoded = encode_categorical_feature(flag4, "flag4", train_ds, False)

all_features = layers.concatenate(
    [
        size_encoded,
        flag1_encoded,
        flag2_encoded,
        flag3_encoded,
        flag4_encoded,
    ]
)
x_layers = layers.Dense(32, activation="relu")(all_features)
x_layers = layers.Dropout(0.5)(x_layers)
output = layers.Dense(1, activation="sigmoid")(x_layers)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=50, validation_data=val_ds)

model.save('./ModelData/')

keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


sample0 = {
    "size": 19.2,
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