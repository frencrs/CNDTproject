import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.layers import IntegerLookup
from keras.layers import Normalization
from keras.layers import StringLookup


input_dataframe = pd.read_csv("TestData3.csv")

val_dataframe = input_dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = input_dataframe.drop(val_dataframe.index)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #ds = ds.shuffle(buffer_avg_len=len(dataframe))
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
avg_len = keras.Input(shape=(1,), name="avg_len")
# Categorical features encoded as integers
num_unique_ports = keras.Input(shape=(1,), name="num_unique_ports")
num_unique_dst = keras.Input(shape=(1,), name="num_unique_dst")
num_unique_src = keras.Input(shape=(1,), name="num_unique_src")
num_unique_ttl = keras.Input(shape=(1,), name="num_unique_ttl")
num_unique_chksum = keras.Input(shape=(1,), name="num_unique_chksum")
packet_count = keras.Input(shape=(1,), name="packet_count")


all_inputs = [
    avg_len,
    num_unique_ports, num_unique_dst, num_unique_src, num_unique_ttl, num_unique_chksum,
    packet_count,
]

avg_len_encoded = encode_numerical_feature(avg_len, "avg_len", train_ds)
num_unique_ports_encoded = encode_numerical_feature(num_unique_ports, "num_unique_ports", train_ds)
num_unique_dst_encoded = encode_numerical_feature(num_unique_dst, "num_unique_dst", train_ds)
num_unique_src_encoded = encode_numerical_feature(num_unique_src, "num_unique_src", train_ds)
num_unique_ttl_encoded = encode_numerical_feature(num_unique_ttl, "num_unique_ttl", train_ds)
num_unique_chksum_encoded = encode_numerical_feature(num_unique_chksum, "num_unique_chksum", train_ds)
packet_count_encoded = encode_numerical_feature(packet_count, "packet_count", train_ds)

all_features = layers.concatenate(
    [
        avg_len_encoded,
        num_unique_ports_encoded,
        num_unique_dst_encoded,
        num_unique_src_encoded,
        num_unique_ttl_encoded,
        num_unique_chksum_encoded,
        packet_count_encoded,
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


#sample0 = {
#    "avg_len": 19.2,
#    "num_unique_ports": 1,
#    "num_unique_dst": 1,
#    "num_unique_src": 0,
#    
#}

#sample1 = {
 #   "avg_len": 3.8,
 #   "num_unique_ports": 0,
 #   "num_unique_dst": 0,
 #   "num_unique_src": 1,
 #   "num_unique_ttl": 0,
#}

#input_dict1 = {name: tf.convert_to_tensor([value]) for name, value in sample0.items()}
#input_dict2 = {name: tf.convert_to_tensor([value]) for name, value in sample1.items()}

#predictions1 = model.predict(input_dict1)
#predictions2 = model.predict(input_dict2)

#print("prediction1", predictions1)
#print("prediction2", predictions2)