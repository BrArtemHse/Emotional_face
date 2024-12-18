import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from learning_model import DataGeneration, create_generation_for_training, validation, create_base_model, create_model

class TestLearningModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 100
        self.img_shape = 128
        self.test_dir = Path("data_set")
        self.num_classes = 8
        self.subset = "training"

    def test_data_generation(self):
        data_gen = DataGeneration()
        self.assertIsInstance(data_gen, tf.keras.preprocessing.image.ImageDataGenerator)

    def test_create_generation_for_training(self):
        train_gen = create_generation_for_training(self.batch_size, self.img_shape, self.test_dir)
        self.assertEqual(train_gen.batch_size, self.batch_size)
        self.assertEqual(train_gen.image_shape, (self.img_shape, self.img_shape, 3))
        self.assertEqual(train_gen.class_mode, "categorical")
        self.assertEqual(train_gen.subset, "training")
        self.subset = "validation"

    def test_validation_data_generation(self):
        val_gen = validation(self.batch_size, self.img_shape, self.test_dir)
        self.assertEqual(val_gen.batch_size, self.batch_size)
        self.assertEqual(val_gen.image_shape, (self.img_shape, self.img_shape, 3))
        self.assertEqual(val_gen.class_mode, "categorical")
        self.assertEqual(val_gen.subset, "validation")

    def test_create_model(self):
        model = create_model(self.num_classes)
        self.assertIsInstance(model, tf.keras.Sequential)
        self.assertEqual(len(model.layers), 5)

if __name__ == '__main__':
    unittest.main()