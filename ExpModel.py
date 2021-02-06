import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, precision_recall_fscore_support

import keras
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input as preprocessVGG
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.inception_v3 import preprocess_input as preprocessInception
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import optimizers

class ExpModel:
	def __init__(self, ModelName, DataSet, x_train, y_train, x_valid, y_valid, x_test, y_test):
		self.ModelName = ModelName.lower()
		self.DataSet = DataSet.lower()
		self.x_train = x_train
		self.y_train = y_train
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.x_test = x_test
		self.y_test = y_test
		self._no_class = self._getNoClasses()
		self._isColored()
		self._datasetParameters()
		self._modelParameters()

	def _getNoClasses(self):
		switcher = {
		'mnist'					: 10,
		'fashionmnist'			: 10,
		'cifar10'				: 10,
		'trafficsign'			: 43,
		'smallnorb'				: 5,
		'shapes3d_floor_hue'	: 10,
		'shapes3d_object_hue'	: 10,
		'shapes3d_orientation'	: 15,
		'shapes3d_scale'		: 8,
		'shapes3d_shape'		: 4
		}
		return switcher.get(self.DataSet, 'Cannot get number of classes')

	def _isColored(self):
		if (self.DataSet == 'mnist') or (self.DataSet == 'fashionmnist') or (self.DataSet == 'smallnorb'):
			self._is_colored = False
		else:
			self._is_colored = True

	def _datasetParameters(self):
		self._IMG_DEPTH = 3
		if self.DataSet == 'smallnorb':
			self._IMG_HEIGHT = 96
			self._IMG_WIDTH = 96
		else:
			self._dimensions()

	def _dimensions(self):
		if (self.ModelName == 'vgg'):
			self._IMG_HEIGHT, self._IMG_WIDTH = 48, 48
		elif self.ModelName == 'inception':
			self._IMG_HEIGHT, self._IMG_WIDTH = 75, 75
		elif self.ModelName == 'resnet':
			self._IMG_HEIGHT, self._IMG_WIDTH = 32, 32

	def _modelParameters(self):
		self._NB_EPOCHS = 20
		self._BATCH_SIZE = 20

	def trainModel(self):
		self._prepareDataset()
		self._buildBaseModel()
		# if self.ModelName == 'vgg':
		# 	self._prepareForVGG()
		self._buildModel()
		self._compileModel()
		self._defineCallbacks()
		self._fitModel()
		self._modelPerformance()
		self._testModel()
		self.results = {'train':[self._acc[-1], self._val_acc[-1], self._loss[-1], self._val_loss[-1]],\
             'test':[self.test_acc, self.test_precision, self.test_recall, self.test_fscore, self.hamming]}
		return self.model, self.history, self.results

	def _prepareDataset(self):
		self.x_train = self._prepareXData(self.x_train)
		self.x_valid = self._prepareXData(self.x_valid)
		self.x_test = self._prepareXData(self.x_test)

		self.y_train = to_categorical(self.y_train)
		self.y_valid = to_categorical(self.y_valid)
		self.y_test = to_categorical(self.y_test)

	def _prepareXData(self, DS):
		height = self._IMG_HEIGHT
		width = self._IMG_WIDTH
		if self.ModelName == 'inception':
			if self._is_colored:
				DS_preprocess = []
			else:
				DS = DS.astype(np.float) / 255.0
				DS_preprocess = np.zeros((DS.shape[0], self._IMG_HEIGHT, self._IMG_WIDTH, 3), dtype=np.float32)
			for i, img in enumerate(DS):
				img_resize = resize(img, (self._IMG_HEIGHT, self._IMG_WIDTH), anti_aliasing=True)
				img_resize = preprocessInception(img_resize).astype(np.float32)
				if self._is_colored:
					DS_preprocess.append(img_resize)
				else:
					DS_preprocess[i] = np.dstack([img_resize, img_resize, img_resize])
			if self._is_colored:
				DS_preprocess = np.array(DS_preprocess)
			return DS_preprocess
		else:
			if (self.DataSet == 'mnist') or (self.DataSet == 'fashionmnist'):
				height, width = 28, 28
			if self._is_colored:
				DS = np.asarray([img_to_array(array_to_img(im, scale=False).resize((self._IMG_HEIGHT, self._IMG_WIDTH))) for im in DS])
			else:
				DS = np.dstack([DS] * 3)
				DS = DS.reshape(-1, height, width, 3)
				DS = np.asarray([img_to_array(array_to_img(im, scale=False).resize((self._IMG_HEIGHT, self._IMG_WIDTH))) for im in DS])
				DS = DS / 255.
				DS = DS.astype('float32')
			if (self.ModelName == 'vgg'):
				DS = preprocessVGG(DS)
			return DS

	# def _prepareXData(self, DS):
	# 	height = self._IMG_HEIGHT
	# 	width = self._IMG_WIDTH
	# 	if (self.DataSet == 'mnist') or (self.DataSet == 'fashionmnist'):
	# 		height, width = 28, 28
	# 	if (self.ModelName == 'vgg') or (self.ModelName == 'resnet'):
	# 		DS = np.dstack([DS] * 3)
	# 		DS = DS.reshape(-1, height, width, 3)
	# 		DS = np.asarray([img_to_array(array_to_img(im, scale=False).resize((self._IMG_HEIGHT, self._IMG_WIDTH))) for im in DS])
	# 		DS = DS / 255.
	# 		DS = DS.astype('float32')
	# 		return DS
	# 	elif self.ModelName == 'inception':
	# 		DS = DS.astype(np.float) / 255.0
	# 		DS_preprocess = np.zeros((DS.shape[0], self._IMG_HEIGHT, self._IMG_WIDTH, 3), dtype=np.float32)
	# 		for i, img in enumerate(DS):
	# 			img_resize = resize(img, (self._IMG_HEIGHT, self._IMG_WIDTH), anti_aliasing=True)
	# 			img_resize = preprocessInception(img_resize).astype(np.float32)
	# 			DS_preprocess[i] = np.dstack([img_resize, img_resize, img_resize])
	# 		return DS_preprocess

	def _buildBaseModel(self):
		INPUT_SHAPE = (self._IMG_HEIGHT, self._IMG_WIDTH, self._IMG_DEPTH)
		if self.ModelName == 'vgg':
			#pooling='max',
			self._baseModel = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
		elif self.ModelName == 'inception':
			self._baseModel = InceptionV3(weights='imagenet', include_top=False,\
					                       pooling='avg', input_shape=INPUT_SHAPE)
		elif self.ModelName == 'resnet':
			self._baseModel = ResNet50(weights='imagenet', include_top=False,\
							            pooling='avg', input_shape=INPUT_SHAPE)

	# def _prepareForVGG(self):
	# 	self.x_train = preprocessVGG(self.x_train)
	# 	self.x_valid = preprocessVGG(self.x_valid)
	# 	self.x_test  = preprocessVGG(self.x_test)

	def _buildModel(self):
		if self.ModelName == 'vgg':
			x = Flatten()(self._baseModel.output)
		elif (self.ModelName == 'inception') or (self.ModelName == 'resnet'):
			x = self._baseModel.output
		x = Dense(1000, activation='relu')(x)
		predictions = Dense(self._no_class, activation='softmax')(x)
		self.model = Model(inputs=self._baseModel.input, outputs=predictions)
		print(f"<<< {self.ModelName.upper()} Model is prepared >>>")
		self.model.summary()
		for layer in self._baseModel.layers:
			layer.trainable = False

	def _compileModel(self):
		self.model.compile(loss='categorical_crossentropy',
						  optimizer=optimizers.Adam(),
						  metrics=['acc'])

	def _defineCallbacks(self):
		from keras import callbacks
		reduce_learning = callbacks.ReduceLROnPlateau(monitor='val_loss',
													factor=0.2,
													patience=2,
													verbose=1,
													mode='auto',
													min_delta=0.0001,
													cooldown=2,
													min_lr=0)

		early_stopping = callbacks.EarlyStopping(monitor='val_loss',
												min_delta=0,
												patience=7,
												verbose=1,
												mode='auto')

		self._callbacks = [reduce_learning, early_stopping]

	def _fitModel(self):
		self.history = self.model.fit(self.x_train,
									self.y_train,
									epochs=self._NB_EPOCHS,
									validation_data=(self.x_valid, self.y_valid),
									callbacks=self._callbacks)

	def _modelPerformance(self):
		self._acc = self.history.history['acc']
		self._val_acc = self.history.history['val_acc']
		self._loss = self.history.history['loss']
		self._val_loss = self.history.history['val_loss']
		print('*********************************************')
		print(f'Model training accuracy: {self._acc[-1]}')
		print(f'Model validation accuracy: {self._val_loss[-1]}')
		print(f'Model training loss: {self._acc[-1]}')
		print(f'Model validation loss: {self._val_loss[-1]}')


	def _testModel(self):
		y_pred = self.model.predict(self.x_test)
		self.test_acc = accuracy_score(self.y_test, y_pred.round(), normalize=True, sample_weight=None)
		self.test_precision, self.test_recall, self.test_fscore, support = precision_recall_fscore_support(self.y_test, y_pred.round(), average='weighted')
		self.hamming = hamming_loss(self.y_test, y_pred.round())
		print('*********************************************')
		print(f'Model testing accuracy: {self.test_acc}')
		print(f'Model testing precision: {self.test_precision}')
		print(f'Model testing recall: {self.test_recall}')
		print(f'Model testing F-score: {self.test_fscore}')
		print(f'Model testing hamming loss: {self.hamming}')
