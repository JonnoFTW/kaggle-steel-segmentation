from datetime import datetime

from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, \
    MaxPooling2D, Input, UpSampling2D, Concatenate, Reshape, Activation, Permute, PReLU
import keras.backend as K
from keras.optimizers import Adam, Adadelta
from keras.models import Model, load_model
from segmentation_models.losses import DiceLoss, CategoricalFocalLoss, JaccardLoss
from segmentation_models.metrics import IOUScore, FScore

# from LRFinder import LRFinder
# from metrics import dice_coef, mean_iou, iou, dice_coef_loss, jaccard_coef_logloss, tversky_loss
import tensorflow as tf
import pickle


def schedule(epoch, old_lr):
    return old_lr * 0.999


class UnetModel:
    def __init__(self, num_filters=32, input_size=(256, 256, 1), num_classes=4, class_weights=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
        self.input_size = input_size
        self.num_filters = num_filters
        self.image_rows = input_size[0]
        self.image_cols = input_size[1]
        self.class_weights = class_weights
        self.built = False
        self.num_classes = num_classes

    def build(self):
        input_size = self.input_size
        num_filters = self.num_filters
        inputs = Input(input_size)

        weight_initializer = 'he_normal'
        activation = 'relu'
        # BLOCK 1
        conv1 = Conv2D(num_filters, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(inputs)
        # conv1 = PReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(num_filters, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv1)
        # conv1 = PReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # also have a resnet here

        # BLOCK 2
        conv2 = Conv2D(num_filters * 2, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(num_filters * 2, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # BLOCK 3
        conv3 = Conv2D(num_filters * 4, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(num_filters * 4, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # BLOCK 4
        conv4 = Conv2D(num_filters * 8, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(num_filters * 8, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

        # BLOCK 5
        conv5 = Conv2D(num_filters * 16, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(num_filters * 16, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv5)
        conv5 = BatchNormalization()(conv5)

        # BLOCK 6
        up6 = Conv2D(num_filters * 8, 2, activation=activation, padding='same', kernel_initializer=weight_initializer)(UpSampling2D(size=(2, 2))(conv5))
        merge6 = Concatenate(axis=3)([conv4, up6])
        conv6 = Conv2D(num_filters * 8, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(num_filters * 8, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(num_filters * 4, 2, activation=activation, padding='same', kernel_initializer=weight_initializer)(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(num_filters * 4, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(num_filters * 4, 3, activation=activation, padding='same',
                       kernel_initializer=weight_initializer)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(num_filters * 2, 2, activation=activation, padding='same', kernel_initializer=weight_initializer)(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(num_filters * 2, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(num_filters * 2, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(num_filters, 2, activation=activation, padding='same', kernel_initializer=weight_initializer)(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(num_filters, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(num_filters, 3, activation=activation, padding='same', kernel_initializer=weight_initializer)(conv9)
        conv9 = BatchNormalization()(conv9)

        conv9 = Conv2D(self.num_classes, kernel_size=(3, 3), activation=activation, padding='same', kernel_initializer='glorot_uniform')(conv9)

        # reshape = Reshape((self.input_size[0] * self.input_size[1], self.num_classes), input_shape=(self.input_size[0], self.input_size[1], self.num_classes))(conv9)

        # conv10 = Conv2D(self.num_classes, (1, 1), activation='sigmoid')(conv9)
        # conv10 = BatchNormalization()(conv10)
        # conv10 = Reshape((self.input_size[0] * self.input_size[1], self.num_classes),
        #                  input_shape=(self.input_size[0], self.input_size[1], self.num_classes))(conv10)
        # permute = Permute((2,1))(reshape)
        output = Activation('softmax')(conv9)
        model = Model(inputs=inputs, outputs=output, name="Unet Multi")

        metrics = [
            'accuracy',
            IOUScore(threshold=0.5),
            FScore(threshold=0.5)
        ]
        # self.loss_func = get_loss()
        dice_loss = DiceLoss(class_weights=self.class_weights)
        jaccard_loss = JaccardLoss()
        focal_loss = CategoricalFocalLoss()
        total_loss = focal_loss  # + jaccard_loss
        self.loss = total_loss
        model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
        self.built = True
        self.model = model
        self.tensorboard = TensorBoard(f'logs/unet_{self.num_filters}_multi_{datetime.now().timestamp()}',
                                       update_freq=128,
                                       write_graph=True)
        self.model_checkpoint = ModelCheckpoint(f'models/unet_{self.num_filters}_multi_{datetime.now().timestamp()}.hdf5',
                                                monitor='loss', verbose=1, save_best_only=True)

    def fit(self, train_gen, batch_size, num_samples, epochs=1):
        if not self.built:
            self.build()
        print(f"Training for {epochs} with steps_per_epoch={num_samples // batch_size}")

        # def find_lr_epoch_end(epoch, logs):
        #     if epoch % 30 == 0:
        #         lrf = LRFinder(self.model)
        #         lrf.find_generator(train_gen, 0.0001, 1, 2, num_samples // batch_size)
        #         new_lr = lrf.get_best_lr(4)
        #         print("New lr:", new_lr)
        #         self.set_lr(new_lr)
        # lrfinder = LambdaCallback(on_epoch_end=find_lr_epoch_end)
        return self.model.fit_generator(train_gen,
                                        callbacks=[
                                            self.model_checkpoint,
                                            self.tensorboard,
                                            # lrfinder
                                        ],
                                        epochs=epochs,
                                        steps_per_epoch=num_samples // batch_size)

    def predict(self, images):
        return self.model.predict(images)

    def save(self):
        fname = f'models/unet_{self.num_filters}_{int(datetime.now().timestamp())}.h5'
        print(f"Saving model as {fname}")
        with open(fname + '.misc.pkl', 'wb') as fh:
            pickle.dump({
                'model_cb': {'filepath': self.model_checkpoint.filepath,
                             'monitor': self.model_checkpoint.monitor,
                             'mode': 'max',
                             'verbose': 1,
                             'save_best_only': True},
                'tb_cb': {'log_dir': self.tensorboard.log_dir,
                          'update_freq': self.tensorboard.update_freq,
                          'write_graph': self.tensorboard.write_graph}
            }, fh)
        return self.model.save(fname)

    def load(self, fname):
        with open(fname + '.misc.pkl', 'rb') as fh:
            obj = pickle.load(fh)
            self.model_checkpoint = ModelCheckpoint(**obj['model_cb'])
            self.tensorboard = TensorBoard(**obj['tb_cb'])
        self.model = load_model(fname, custom_objects={

        })

    def set_lr(self, new_lr):
        return K.set_value(self.model.optimizer.lr, new_lr)
