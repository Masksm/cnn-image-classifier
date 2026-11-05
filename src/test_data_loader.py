from data_loader import load_datasets


train_dataset, validation_dataset, test_dataset = load_datasets()


print("\n" + "=" * 60)
print("DATASET LOADING TEST")
print("=" * 60)

print("\nClass names:")
print(train_dataset.class_names)

print("\nNumber of classes:")
print(len(train_dataset.class_names))

print("\nImage shape:")
for images, labels in train_dataset.take(1):
    print(images.shape)
    print(labels.shape)

print("\nDataset loading successful.")