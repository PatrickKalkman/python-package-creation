# Tensorflow_Helpers

Tensorflow_Helpers is a Python library with helper function for TensorFlow.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tensorflow_helpers.

```bash
pip install tensorflow_helpers
```

## Usage

```python
from tensorflow_helpers.augmentation import MixUpImageDataGenerator

inp_train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=260,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.25
    )

train_data = pd.read_csv('./data/merged_data.csv')
train_data['label'] = train_data['label'].astype(str)
Y = train_data[['label']]

train_iterator = MixupImageDataGenerator(
    generator=inp_train_gen,
    directory='./data/train_images/',
    img_width=IMG_SIZE[0],
    img_height=IMG_SIZE[1],
    batch_size=BATCH_SIZE,
    subset='training'
)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)