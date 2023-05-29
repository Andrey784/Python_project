from PyQt6.QtCore import QDateTime, QEvent, QObject, Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QMainWindow, QScrollArea,QFileDialog, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QStandardItemModel
from PyQt6 import QtGui
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing import image

QtGui.QImageReader.setAllocationLimit(0)

QtGui.QImageReader.setAllocationLimit(0)

  
class CheckableComboBox(QComboBox):
    # Класс для создания списка CheckBoxы
    # constructor
    def __init__(self, parent = None):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QStandardItemModel(self))
        self._changed = False
    def build(self, state = 'Unchecked'):
        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            if state == 'Unchecked':
                item.setCheckState(Qt.CheckState.Unchecked)
            if state == 'Checked':
                item.setCheckState(Qt.CheckState.Checked)
    def hidePopup(self):
        if not self._changed:
            super(CheckableComboBox, self).hidePopup()
        self._changed = False
    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.CheckState.Checked
    def setItemChecked(self, index, checked=True):
        item = self.model().item(index, self.modelColumn())
        if checked:
            item.setCheckState(Qt.CheckState.Checked)
        else:
            item.setCheckState(Qt.CheckState.Unchecked)
    # when any item get pressed
    def handleItemPressed(self, index):
        # getting the item
        item = self.model().itemFromIndex(index)
        if index.row()==0:
            if item.checkState()==Qt.CheckState.Checked:
                self.build(state='Unchecked')
            else:
                self.build(state='Checked')
        else:
            # checking if item is checked
            if item.checkState() == Qt.CheckState.Checked:
                # making it unchecked
                item.setCheckState(Qt.CheckState.Unchecked)
            # if not checked
            else:
                # making the item checked
                item.setCheckState(Qt.CheckState.Checked)
        self._changed = True

    def get_selected(self):
        model = self.model()
        select_layer = []
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                select_layer.append(item.text())
        return select_layer

class MainWindow(QMainWindow):
    # Главное окно приложения
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.resize(1200,600)
        self.tab_widget = MainTabWindow()
        self.setWindowTitle('Neuron work')
        self.setCentralWidget(self.tab_widget)
        self.set_menu_bar()
        self.show()

    def set_menu_bar(self):
        # Функция отвечает за конструкцию и функциональность меню бара
        menubar = self.menuBar()

        font = menubar.font()
        font.setPixelSize(13)
        menubar.setFont(font)

        menu_file = menubar.addMenu('&File')
        menu_open = menu_file.addAction('&Open')
        menu_open.triggered.connect(self.load_file)

    def load_file(self):
        # Выбор файла модели для загрузки
        file = QFileDialog.getOpenFileName(self, filter='*.h5')[0]
        if file:
           # model = tf.keras.models.load_model(file)
            self.tab_widget.show_info_model(file)

class TiledGradients(tf.Module):
            # Класс отвечающий за создания изображения Deep Dream модели
            @classmethod
            def random_roll(cls, img, maxroll):
                # Увеличение изображения за счёт добавления её частей.
                shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
                img_rolled = tf.roll(img, shift=shift, axis=[0,1])
                return shift, img_rolled
            
            @classmethod
            def calc_loss(cls, img, model, layer, filters = None):
                # Подчёт функции потерь на выбпанном слое и его фильтрах
                img_batch = tf.expand_dims(img, axis=0)
                layer_activations = model(img_batch)[layer]
                if filters:
                    losses = []
                    for i_filter in filters:
                        losses.append(layer_activations[:,:,i_filter])
                    return  tf.reduce_sum(losses)
                return tf.reduce_mean(layer_activations)
            
            def __init__(self, model, layer, filters):
                self.model = model
                self.layer=  layer
                self.filters = filters

            @tf.function(
                input_signature=(
                    tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
                    tf.TensorSpec(shape=[2], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),)
            )
            def __call__(self, img, img_size, tile_size=512):
                # Процесс создания изображения Deep Dream
                shift, img_rolled = self.random_roll(img, tile_size)

                # Initialize the image gradients to zero.
                gradients = tf.zeros_like(img_rolled)

                # Skip the last tile, unless there's only one tile.
                xs = tf.range(0, img_size[1], tile_size)[:-1]
                if not tf.cast(len(xs), bool):
                    xs = tf.constant([0])
                ys = tf.range(0, img_size[0], tile_size)[:-1]
                if not tf.cast(len(ys), bool):
                    ys = tf.constant([0])

                for x in xs:
                    for y in ys:
                        # Calculate the gradients for this tile.
                        with tf.GradientTape() as tape:
                            # This needs gradients relative to `img_rolled`.
                            # `GradientTape` only watches `tf.Variable`s by default.
                            tape.watch(img_rolled)
                            # Extract a tile out of the image.
                            img_tile = img_rolled[y:y+tile_size, x:x+tile_size]
                            loss = self.calc_loss(img_tile, self.model, self.layer, filters = self.filters)

                        # Update the image gradients for this tile.
                        gradients = gradients + tape.gradient(loss, img_rolled)

                # Undo the random shift applied to the image and its gradients.
                gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

                # Normalize the gradients.
                gradients /= tf.math.reduce_std(gradients) + 1e-8 

                return gradients
            

class MainTabWindow(QTabWidget):
    # Класс отвечает за конструкцию и функционал вкладок в главном окне
    def __init__(self):
        super(MainTabWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        # Общий интерфейс
        self.setStyleSheet('''
        QScrollArea, QLabel{
            background-color: #ffffff;
        }
        QPushButton{
            font-size:18px;
        }
    ''')
        def init_tab1_describe():
            # Вкладка общей информации о модели
            self.central_layout = QHBoxLayout()

            #Row1     
            self.tab_text_image_desqribe = QTabWidget()
            self.scroll_area_image = QScrollArea()
            self.tab_text_image_desqribe.addTab(self.scroll_area_image, 'Image')
            self.label_img = QLabel()
            self.scroll_area_image.setWidget(self.label_img)
            self.scroll_area_image.setWidgetResizable(True)
            self.label_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
            self.scroll_area_text = QScrollArea()
            self.tab_text_image_desqribe.addTab(self.scroll_area_text, 'Text')
            self.label_text = QLabel()
            self.scroll_area_text.setWidget(self.label_text)
            self.scroll_area_text.setWidgetResizable(True)
            self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.central_layout.addWidget(self.tab_text_image_desqribe,1)

            #Row2
            self.frame = QFrame()
            self.scroll_area_layers = QScrollArea()
            self.scroll_area_layers.setWidgetResizable(True)
            self.scroll_area_layers.setWidget(self.frame)

            self.btn_layout = QVBoxLayout(self.frame)
            self.central_layout.addWidget(self.scroll_area_layers,1)

            #Row3
            self.box_layout = QGroupBox('Веса слоя')
            self.scroll_area_graph = QScrollArea()
            self.right_row_layout = QVBoxLayout(self.box_layout)
            self.buttonConvFilters = QComboBox()
            self.buttonConvFilters.hide()
            self.graphic_weight = QLabel()
            self.graphic_bias = QLabel()
            self.right_row_layout.addWidget(self.buttonConvFilters)
            self.right_row_layout.addWidget(self.graphic_weight)
            self.right_row_layout.addWidget(self.graphic_bias)
            self.box_layout.setLayout(self.right_row_layout)

            self.graphic_weight.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.graphic_bias.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll_area_graph.setWidget(self.box_layout)
            self.scroll_area_graph.setWidgetResizable(True)
            self.central_layout.addWidget(self.scroll_area_graph, 1)

            self.tab1_describe.setLayout(self.central_layout)

        def init_tab_work_con():
            # Вкладка визуализации работы свёрточных слоёв
            self.central_vlayout_conv_work = QVBoxLayout()
            self.central_hlayout_conv_work = QHBoxLayout()

            self.button_load_image = QPushButton('load_image')
            self.button_load_image.clicked.connect(self.load_image_in_model)
            self.button_load_image.setStyleSheet("max-width:100px; font-size:12px")
            self.central_vlayout_conv_work.addWidget(self.button_load_image)

            #row1
            self.frame_conv = QFrame()
            self.scroll_area_layers_conv = QScrollArea()
            self.scroll_area_layers_conv.setWidgetResizable(True)
            self.scroll_area_layers_conv.setWidget(self.frame_conv)

            self.btn_layout_conv = QVBoxLayout(self.frame_conv)
            
            self.central_hlayout_conv_work.addWidget(self.scroll_area_layers_conv,1)

            #row2
            self.scroll_area_images_work_conv = QScrollArea()
            self.label_images_work_conv = QLabel()
            self.scroll_area_images_work_conv.setWidget(self.label_images_work_conv)
            self.scroll_area_images_work_conv.setWidgetResizable(True)
            self.label_images_work_conv.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.central_hlayout_conv_work.addWidget(self.scroll_area_images_work_conv,2)

            self.central_vlayout_conv_work.addLayout(self.central_hlayout_conv_work)
            self.tab2_conv_activation.setLayout(self.central_vlayout_conv_work)

        def init_tap_deep_dream():
            # Вкладка для демонстрации Deep Dream
            self.central_vlayout_conv_deep_dream = QVBoxLayout()
            self.central_hlayout_conv_deep_dream = QHBoxLayout()

            self.layout_for_button_deep_dream = QHBoxLayout()
            self.button_load_image_for_deep_dream = QPushButton('Load_image')
            self.button_load_image_for_deep_dream.clicked.connect(self.load_image_in_model)
            self.button_load_image_for_deep_dream.setStyleSheet("max-width:100px; font-size:12px")
            self.button_delete_image_for_deep_dream = QPushButton('Delete_image')
            self.button_delete_image_for_deep_dream.clicked.connect(self.delete_image)
            self.button_delete_image_for_deep_dream.setStyleSheet("max-width:100px; font-size:12px")
            self.layout_for_button_deep_dream.addWidget(self.button_load_image_for_deep_dream)
            self.layout_for_button_deep_dream.addWidget(self.button_delete_image_for_deep_dream)
            
            #row1
            self.layout_first_row_deep_dream = QVBoxLayout()
            self.frame_conv_deep_dream = QFrame()
            self.scroll_area_layers_conv_deep_dream = QScrollArea()
            self.scroll_area_layers_conv_deep_dream.setWidgetResizable(True)
            self.scroll_area_layers_conv_deep_dream.setWidget(self.frame_conv_deep_dream)

            self.btn_layout_conv_deep_dream = QVBoxLayout(self.frame_conv_deep_dream)
            
            self.layout_first_row_deep_dream.addLayout(self.layout_for_button_deep_dream)
            self.layout_first_row_deep_dream.addWidget(self.scroll_area_layers_conv_deep_dream)

            self.central_hlayout_conv_deep_dream.addLayout(self.layout_first_row_deep_dream,1)

            #Row2
            self.layout_output_deep_dream = QGroupBox('Deep Dream')
            self.scroll_area_dream = QScrollArea()
            self.row_layout_dream = QVBoxLayout(self.layout_output_deep_dream)
            self.layout_button_dream = QHBoxLayout()
            self.edit_size_step = QLineEdit('0.01')
            self.button_start_deep = QPushButton('Start')
            self.button_start_deep.clicked.connect(self.run_deep_dream)
            self.edit_size_step.setStyleSheet("max-width:100px; max-height:25px")
            self.button_start_deep.setStyleSheet("max-width:100px; max-height:20px")
            self.combo_box = CheckableComboBox(self)
            #self.combo_box.hide()
            self.layout_button_dream.addWidget(self.edit_size_step)
            self.layout_button_dream.addWidget(self.combo_box)
            self.layout_button_dream.addWidget(self.button_start_deep)
            self.label_images_deep_dream = QLabel()
            self.row_layout_dream.addLayout(self.layout_button_dream,1)
            self.row_layout_dream.addWidget(self.label_images_deep_dream,1)
            self.layout_output_deep_dream.setLayout(self.row_layout_dream)

            self.label_images_deep_dream.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll_area_dream.setWidget(self.layout_output_deep_dream)
            self.scroll_area_dream.setWidgetResizable(True)
            self.central_vlayout_conv_deep_dream.addWidget(self.scroll_area_dream, 1)
            self.central_hlayout_conv_deep_dream.addLayout(self.central_vlayout_conv_deep_dream,2)
            self.tab3_deep_dream.setLayout(self.central_hlayout_conv_deep_dream)

        self.load_image = None
        self.load_image_deep = None
        self.tab1_describe = QWidget()
        self.tab2_conv_activation = QWidget()
        self.tab3_deep_dream = QWidget()

        self.addTab(self.tab1_describe,"Общая информация")
        self.addTab(self.tab2_conv_activation,"Работа свёрточных слоёв")
        self.addTab(self.tab3_deep_dream, 'Deep Dream')
        init_tab1_describe()
        init_tab_work_con()
        init_tap_deep_dream()

    def show_text_desqribe_model(self, path):
        with open(path, 'r') as f:
            text = '\n'.join(f.readlines())
        self.label_text.setText(text)
        self.label_text.setFont(QFont('Aria', 10, weight = 30))

    def load_model(self, path):
        try:
            import keras.applications as modells
            #self.model = tf.keras.models.load_model(path)
            self.model = modells.inception_v3.InceptionV3(weights='imagenet')
        except:
            raise 'Error'
        
    def show_info_model(self, path):
        self.load_model(path)
        self.load_info_model()
        self.show_img(r'tmp\plot_model.jpeg')
        self.show_text_desqribe_model(r'tmp/summary.txt')
        self.show_layers()
        self.get_layers_output_model()
        self.show_layers_conv()
        self.show_layers_conv_deep_dream()
        
    def load_info_model(self):
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        self.name_layers = dict(map(lambda x: (x.name, x), self.model.layers))
        tf.keras.utils.plot_model(self.model, to_file=r'tmp\plot_model.jpeg', dpi=115)
        with open(r'tmp/summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n\n'))

    def delete_image(self):
        self.load_image_deep = None

    def load_image_in_model(self):
        send = self.sender()
        path_image = QFileDialog.getOpenFileName(self, filter='Images (*.png *.jpg *.jpeg )')[0]
        if path_image:
            new_image = image.load_img(path_image)
            new_image = image.img_to_array(new_image)
            new_image = (new_image / 255)*2-1
        if send.text()=='load_image':
            self.load_image = new_image
        if send.text()=='Load_image':
            self.load_image_deep = new_image
            
    def show_layers(self):
        for i in reversed(range(self.btn_layout.count())): 
            self.btn_layout.itemAt(i).widget().setParent(None)
        for i in self.model.layers:
            btn = QPushButton(i.name)
            btn.clicked.connect(self.show_weights)
            self.btn_layout.addWidget(btn)

    def show_layers_conv(self):
        for i in reversed(range(self.btn_layout_conv.count())): 
            self.btn_layout_conv.itemAt(i).widget().setParent(None)
        for i in self.model.layers:
            if i.name.startswith('conv'):
                btn_conv= QPushButton(i.name)
                btn_conv.clicked.connect(self.show_work_conv)
                self.btn_layout_conv.addWidget(btn_conv)
                
    def show_layers_conv_deep_dream(self):
        for i in reversed(range(self.btn_layout_conv_deep_dream.count())): 
            self.btn_layout_conv_deep_dream.itemAt(i).widget().setParent(None)
        for i in self.model.layers:
            if i.name:
                btn_conv= QPushButton(i.name)
                btn_conv.clicked.connect(self.show_deep_dream)
                self.btn_layout_conv_deep_dream.addWidget(btn_conv)

    @classmethod
    def upload_graphics(cls, name_layer, weights = None, bias = None):
        if weights is not None:
            plt.figure(figsize=(4,4))
            sns.histplot(np.ravel(weights)).set_title(name_layer)
            plt.xlabel("weights")
            plt.ylabel(None)
            plt.savefig('tmp/weights_graph.png', dpi = 90, bbox_inches='tight')

        if bias is not None:
            plt.figure(figsize=(4,4))
            sns.histplot(np.ravel(bias)).set_title(name_layer)
            plt.xlabel("bias")
            plt.ylabel(None)
            plt.savefig('tmp/bias_graph.png', dpi = 90, bbox_inches='tight')

    def run_deep_dream_with_octaves(self, img, get_tiled_gradients, steps_per_octave=50, step_size=0.01, 
                                octaves=range(-2,3), octave_scale=1.3):
        def deprocess(img):
            img = 255*(img + 1.0)/2.0
            return tf.cast(img, tf.uint8)
        
        base_shape = tf.shape(img)
        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)
        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
            new_size = tf.cast(new_size, tf.int32)
            img = tf.image.resize(img, new_size)

            for step in range(steps_per_octave):
                gradients = get_tiled_gradients(img, new_size)
                img = img + gradients*step_size
                img = tf.clip_by_value(img, -1, 1)
                if step % 10 == 0:
                    print ("Octave {}, Step {}".format(octave, step))
                    
        result = deprocess(img)
        return result
    
    def run_deep_dream(self):
        self.combo_box.get_selected()
        filters = []
        for item in self.combo_box.get_selected():
            for word in item.split():
                if word.isdigit():
                    filters.append(int(word))

        size_image = list(self.name_layers.values())[0].input.shape[1:-1]
        if self.load_image_deep is None:
            self.load_image_deep = np.clip(np.random.normal(0, 1, size = (size_image[0], size_image[1], 3)), -1, 1)
            
        self.load_image_deep =  image.smart_resize(self.load_image_deep, size_image)
        self.load_image_deep =  self.load_image_deep.reshape((-1, size_image[0], size_image[1], 3))
        get_tiled_gradients = TiledGradients(self.layers_output_model, layer = self.index_layer, filters = filters)
        step_size = float(self.edit_size_step.text())
       
        new_img = self.run_deep_dream_with_octaves(img = self.load_image_deep[0], get_tiled_gradients= get_tiled_gradients, step_size=step_size)
        plt.imshow(new_img)
        plt.savefig(f'tmp/deep_dream.jpg', dpi = 120, bbox_inches='tight')
        pixmap = QPixmap(f'tmp/deep_dream.jpg')
        pixmap = pixmap.scaledToWidth(int(self.label_images_deep_dream.size().width()*0.85))
        self.label_images_deep_dream.clear()
        self.label_images_deep_dream.setPixmap(pixmap)

    def show_deep_dream(self):
        sender = self.sender()
        name_button = sender.text()
        self.layer_for_deep_dream = self.name_layers[name_button]
        self.index_layer = list(self.name_layers.keys()).index(name_button)
        self.combo_box.clear()
        if len(self.layer_for_deep_dream.weights)!=0:
            self.combo_box.addItems(['Все'] + [f'Фильтр - {i}' for i in range(self.layer_for_deep_dream.weights[0].shape[-1])])
        self.combo_box.build()
        
    def show_weights(self):
        sender = self.sender()
        name_button = sender.text()
        layer = self.name_layers[name_button]
        if len(layer.weights)==0:
            print('Слой не имеет параметров')

        else:
            if len(layer.weights)==1:
                self.weights_layer, self.bias_layer = layer.weights[0].numpy(), None
            else:
                self.weights_layer, self.bias_layer = layer.weights[0].numpy(), layer.weights[1].numpy()

            self.upload_graphics(name_button, self.weights_layer, self.bias_layer)

            if name_button.startswith('conv'):
                self.buttonConvFilters.blockSignals(True)
                self.buttonConvFilters.clear()
                self.buttonConvFilters.show()
                self.buttonConvFilters.addItems(['Все'] + [f'Фильтр - {i} Канал - {j}' for i in range(self.weights_layer.shape[-1]) for j in range(self.weights_layer.shape[-2])])
                self.buttonConvFilters.blockSignals(False)
                self.buttonConvFilters.currentTextChanged.connect(self.show_weight_conv_in_matrix)
            else:
                self.buttonConvFilters.hide()
            self.show_graphics()
 
    def show_graphics(self):
        if self.weights_layer is not None:
            self.graphic_weight.clear()
            pixmap_weight = QPixmap('tmp/weights_graph.png')
            self.graphic_weight.setPixmap(pixmap_weight)
        if self.bias_layer is not None:
            self.graphic_bias.clear()
            pixmap_bias = QPixmap('tmp/bias_graph.png')
            self.graphic_bias.setPixmap(pixmap_bias)

    def show_weight_conv_in_matrix(self):
        item = self.buttonConvFilters.currentText()
        weights_conv = np.round(self.weights_layer,4) if self.weights_layer is not None else None
        bias_conv = np.round(self.bias_layer,4) if self.bias_layer is not None else None
        if item == 'Все':
            self.show_graphics()
        else:
            i_filter, i_channel = [int(i) for i in item.split() if i.isdigit()]
            self.graphic_weight.setText('\n'.join([' '.join(i) for i in weights_conv[:,:,i_channel,i_filter].astype('str')]))
            self.graphic_weight.setFont(QFont('Aria', 18, weight = 10))
            if bias_conv is not None:
                self.graphic_bias.setText(str(bias_conv[i_filter]))
                self.graphic_bias.setFont(QFont('Aria', 18, weight = 10))
        
    def show_img(self, path):
        self.pixmap = QPixmap(path)
        self.pixmap = self.pixmap.scaledToWidth(int(self.pixmap.size().width() * (self.label_img.size().width()/self.pixmap.size().width())**0.35))
        self.label_img.clear()
        self.label_img.setPixmap(self.pixmap)

    @classmethod
    def get_output_layer(cls, model, layer, image):
        activations = model.predict(image)[layer][0]
        return activations.reshape(activations.shape[-2],activations.shape[-2], -1)
    
    def get_layers_output_model(self):
        layers = [layer.output for layer in self.model.layers]
        self.layers_output_model = tf.keras.models.Model(self.model.input, layers)

    @classmethod
    def decode_layer(cls, layer, activation = 'relu'):
        layer1 = layer.copy()
        if activation == 'relu':
            if np.max(layer)!=0:
                layer1 = (layer-np.min(layer))/np.max(layer)
        if activation == 'tanh':
            layer1 = (layer + 1)/2
        return layer1

    def upload_image_work_conv(self, i_layer, rows = 4, cols = 16, cmap = None):
        size = self.layers_output_model.layers[i_layer].output.shape[1]
        all_layer = self.get_output_layer(self.layers_output_model, i_layer, self.load_image)
        images_per_row = 4 if all_layer.shape[-1]>3 else 1 
        n_cols = all_layer.shape[-1]//images_per_row if all_layer.shape[-1]>3 else 1
        if n_cols == 1 and images_per_row == 1:
            display_grid = np.zeros((size * n_cols, images_per_row * size, 3))
        else:
            display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                if n_cols == 1 and images_per_row ==1:
                    layer = all_layer[:, :, :]
                    layer = self.decode_layer(layer, activation='tanh')
                    display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size, :3] = layer.reshape(size,size,3)
                else:
                    layer = all_layer[:, :, col * rows + row]
                    layer = self.decode_layer(layer)
                    display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = layer.reshape(size,size)
        scale = 1.5 / size
        plt.figure(figsize=(scale * display_grid.shape[1]*2, scale * display_grid.shape[0]*2))
        plt.grid(False)
        plt.imshow(display_grid,  cmap = cmap)
        plt.savefig('tmp/conv_work.jpg', dpi = 100, bbox_inches='tight')

    def show_work_conv(self):
        if self.load_image is None:
            print('Load image for continue!')
            return
        sender = self.sender()
        name_button = sender.text()
        size_image = list(self.name_layers.values())[0].input.shape[1:-1]
        self.load_image =  image.smart_resize(self.load_image, size_image)
        self.load_image =  self.load_image.reshape((-1, size_image[0], size_image[1], 3))
        layer = self.name_layers[name_button]
        self.upload_image_work_conv(self.model.layers.index(layer))
        pixmap = QPixmap('tmp/conv_work.jpg')
        pixmap = pixmap.scaledToWidth(int(self.label_images_work_conv.size().width()*0.95))
        self.label_images_work_conv.clear()
        self.label_images_work_conv.setPixmap(pixmap)

def run_windor():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
run_windor()