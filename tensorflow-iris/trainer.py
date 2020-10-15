import tensorflow as tf

criteria = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(model, X, Y):
    with tf.GradientTape() as tape:
        prediction = model(X, training=True)
        loss = criteria(Y, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(Y, prediction)


@tf.function
def test_step(model, X, Y):
    predictions = model(X, training=False)
    loss = criteria(Y, predictions)

    test_loss(loss)
    test_accuracy(Y, predictions)


def train(model, dataset, epochs):
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # train_step(model, data, target)
        # test_step(model, data, target)

        for data, target in dataset:
            train_step(model, data, target)

        for data, target in dataset:
            test_step(model, data, target)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result(),
            test_loss.result(),
            test_accuracy.result()
        ))