def train_model(framework, model, train_loader, test_loader, device, epochs, lr):
    if framework == 'tensorflow':
        import tensorflow as tf
        
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)
            test_accuracy(labels, predictions)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for epoch in range(epochs):
            for images, labels in train_loader:
                train_step(images, labels)

            for test_images, test_labels in test_loader:
                test_step(test_images, test_labels)

            print(f'Epoch {epoch + 1}, '
                  f'Loss: {train_loss.result()}, '
                  f'Accuracy: {train_accuracy.result() * 100}, '
                  f'Test Loss: {test_loss.result()}, '
                  f'Test Accuracy: {test_accuracy.result() * 100}')

    elif framework == 'pytorch':
        import torch
        import torch.optim as optim
        import torch.nn.functional as F

        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print(f'Epoch {epoch + 1}: Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
                  f'({100. * correct / len(test_loader.dataset):.2f}%)')

