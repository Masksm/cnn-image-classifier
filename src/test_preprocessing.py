from data_loader import load_datasets


train_dataset, validation_dataset, test_dataset = load_datasets()


for images, labels in train_dataset.take(1):

    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)

    print("\nImage shape:")
    print(images.shape)

    print("\nData type:")
    print(images.dtype)

    print("\nMinimum pixel value:")
    print(images.numpy().min())

    print("\nMaximum pixel value:")
    print(images.numpy().max())

    print("\nFirst image, first pixel:")
    print(images[0, 0, 0].numpy())

    print("\nFirst 5 labels:")
    print(labels[:5].numpy())