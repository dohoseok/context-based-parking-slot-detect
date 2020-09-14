import tensorflow as tf


def model_mobilenetv2(input_tensor, train_lambda=0.1):
    model_input = tf.keras.layers.Input(tensor=input_tensor)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=model_input, pooling='avg')

    # for parking space type
    _ = tf.keras.layers.Dense(units=128, activation='relu')(base_model.output)
    type_output = tf.keras.layers.Dense(units=4, activation='softmax', name='type_output')(_)

    # for parking line angle
    _ = tf.keras.layers.Dense(units=128, activation='relu')(base_model.output)
    angle_output = tf.keras.layers.Dense(units=1, activation='sigmoid', name='angle_output')(_)

    model = tf.keras.models.Model(inputs=model_input, outputs=[type_output, angle_output])

    for layer in model.layers:
        layer.trainable = True

    opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt,
                  loss={'type_output': 'sparse_categorical_crossentropy', 'angle_output': 'mse'},
                  loss_weights={'type_output': train_lambda, 'angle_output': (1-train_lambda)},
                  metrics={'type_output': 'accuracy', 'angle_output': 'mae'})

    return model