from pathlib import Path
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Input
from tensorflow import keras

from sklearn import metrics
from tqdm import tqdm

import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
import warnings

from IPython import embed

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [(str(i)) if i >= 10 else f"0{i}" for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}

def fit_model(model_name: str, data_path: str, batch_size=32, width=80, height=80) -> None:       
    input_tensor = Input(shape=(width, height, 3))

    data_path = Path(data_path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(width, height),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(width, height),
        batch_size=batch_size)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    class_names = train_ds.class_names
    print(class_names)

    model_kwargs = {'input_tensor': input_tensor, 'weights': 'imagenet', 'include_top': False}

    model = InceptionV3(**model_kwargs)

    if model_name.lower() == 'vgg16':
        model = VGG16(**model_kwargs)
    elif model_name.lower() == 'resnet50':
        model = ResNet50(**model_kwargs)

    model.trainable = False

    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = model(input_tensor, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(input_tensor, outputs)

    model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])

    model.fit(train_ds, validation_data=val_ds, epochs=10, use_multiprocessing=True, workers=4)

    model.save(f'{model_name}.h5')

def model_predict(model, dataset, sigmoid=False):
    y_pred = np.array(list())
    y_true =  np.array(list())

    for x, y in tqdm(dataset):
        y_pred_tmp = model.predict_on_batch(x).flatten()

        if sigmoid:
            predictions = tf.nn.sigmoid(y_pred_tmp)
            predictions = tf.where(predictions < 0.5, 0, 1)
            y_pred_tmp = predictions.numpy()

        y_pred = np.concatenate([y_pred, y_pred_tmp], axis=None)
        y_true = np.concatenate([y_true, y.numpy()], axis=None)

    return {'y_true': y_true, 'y_pred': y_pred}

def plot_roc(model_path: str, data_path: str, width=80, height=80, batch_size=32) -> None:    
    # Loading the models
    models = dict()
    for model_path in Path(model_path).iterdir():
        name = model_path.stem
        models[name] = keras.models.load_model(model_path)
        print('Loaded', name)

    # Preparing the dataset
    data_path = Path(data_path)
    dataset = tf.keras.utils.image_dataset_from_directory(
                data_path,
                class_names=['attack', 'normal'],
                seed=123,
                image_size=(width, height),
                batch_size=batch_size)

    # Predicting values
    results = dict()
    for name, model in models.items():
        print("Started predicting values for", name)
        results[name] = model_predict(model, dataset)

        print("Some metrics:")
        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(results[name]['y_pred'])
        predictions = tf.where(predictions < 0.5, 0, 1)

        cm = metrics.confusion_matrix(results[name]['y_true'], predictions.numpy())
        tn, fp, fn, tp = cm.ravel()

        print('\tFPR:', fp  / (fn + tp))
        print('\tFNR:', fp / (fp + tn))

    # Plotting ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    colors = ['red', 'blue', 'black']

    for i, (model, model_metrics) in enumerate(results.items()):
        labels = model_metrics['y_true'].reshape(-1, 1)
        pred = model_metrics['y_pred'].reshape(-1, 1)

        fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=None)
        roc_auc = metrics.auc(fpr, tpr)
        print(model, 'AUC:', roc_auc)

        lw = 2
        ax.plot(fpr, tpr, color=colors[i], lw=lw, label=model, linestyle='dotted')

        if i == 0:
            ax.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
            
        ax.set(xlim=[-0.05, 1.05], xlabel='FPR')
        ax.set(ylim=[-0.05, 1.05], ylabel='TPR')
        ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(.5, 1.16))
    fig.savefig('roc.jpg', format='jpg', dpi=210)

def error_rate(model_name: str, data_path: str, batch_size=32, width=80, height=80) -> None:
    model_path = Path('data/models') / model_name
    if model_path in list(Path('data/models').glob('*.h5')):
        model = keras.models.load_model(model_path)
    else:
        raise ValueError("File '%s' not found" % model_name)

    data_path = Path(data_path)

    columns = ['year', 'month', 'tp', 'tn', 'fp', 'fn', 'fpr', 'fnr']
    df = pd.DataFrame(columns=columns)

    for month_path in data_path.iterdir():
        dataset = tf.keras.utils.image_dataset_from_directory(
                    month_path,
                    seed=123,
                    image_size=(width, height),
                    batch_size=batch_size)

        result = model_predict(model, dataset, sigmoid=True)

        # embed()

        cm = metrics.confusion_matrix(result['y_true'], result['y_pred'])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fn + tp)
        fnr = fp / (fp + tn)

        print('Month:', month_path.name)
        print('\tFPR:', fpr)
        print('\tFNR:', fnr)
        
        data = {
            'year': month_path.parent.name,
            'month': month_path.name,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'fpr': fpr,
            'fnr': fnr
        }
        df = df.append(data, ignore_index=True)
        df.to_csv('%s.csv' % Path(model_name).stem, index=False)

def plot_error_rate(model_name: str) -> None:
    model_path = Path(model_name)

    df = pd.read_csv(model_path)

    df = df.sort_values('month', ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    df['fnr'] = df['fn'] / (df['fn'] + df['tp']) * 100
    df['fpr'] = df['fp'] / (df['fp'] + df['tn']) * 100

    X, Y1, Y2 = list(list(MONTHS.values())), df['fnr'], df['fpr']

    ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
    ax.plot(X, Y2, label='FP', marker='o', ms=9, linestyle='dotted', fillstyle='none', color='black')
    ax.set(xticks=X, xlim=(-1, 12), xlabel='Month')
    ax.tick_params(axis='x', rotation=60)
    ax.set(ylim=(-10, 100), ylabel='Error Rate (%)')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.savefig('%s.png' % model_path.stem, format='png', dpi=210, transparent=True)

def main() -> None:
    data_path = 'data/image/2016/'
    # fit_model(model_name='vgg16', data_path=data_path)

    # plot_roc(model_path='data/models', data_path=data_path)

    # error_rate('resnet50.h5', data_path=data_path, batch_size=64)
    
    plot_error_rate('resnet50.csv')

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    from tensorflow.python.keras import backend as K
    K.set_session(sess)

    warnings.simplefilter('ignore')

    # Font setup
    fonts_path = ['/usr/local/share/fonts/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

    plt.rc('font', family='Palatino', size=20)

    main()