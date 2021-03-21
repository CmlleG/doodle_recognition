from doodle_recognition.params import BUCKET_NAME, BUCKET_FOLDER, CLASSES, NUM_CLASSES, STORAGE_LOCATION, URL_FOLDER, NUM_CLASSES_TEST, CLASSES_TEST
from doodle_recognition.data import create_df, create_train_test_val, reshape_X, save_df_to_gcp
from doodle_recognition.model import init_model, initialize_model, save_model_to_gcp, save_image_to_gcp, save_classification_report_gcp

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

class Trainer(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.name = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = initialize_model ()

    def fit_model(self):
        es = EarlyStopping(patience=2, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train,
                                validation_split= 0.10,
                                epochs=100,
                                batch_size=32,
                                verbose=1,
                                callbacks=[es])

    def evaluate_model(self):
        # Use history to fetch validation score on last epoch
        self.score_name =list(self.history.history.keys())[1]
        self.val_score_name =list(self.history.history.keys())[-1]
        self.score = round(self.history.history[self.val_score_name][-1]*100,2)

        print(f'{self.score_name} = {self.score}')

    def get_classification_report(self):
        # Use history to fetch validation score on last epoch
        test_predictions = np.argmax(self.history.predict(self.X_test), axis=-1)
        report = classification_report(self.y_test,test_predictions)
        print(report)
        save_classification_report_gcp(report)

    def save_fig(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='train')
        ax1.plot(self.history.history['val_loss'], label='val')
        # ax1.set_ylim(0., 2.2)
        ax1.set_title('loss')
        ax1.legend()

        ax2.plot(self.history.history[self.score_name], label=f'train {self.score_name}' )
        ax2.plot(self.history.history[self.val_score_name], label=f'val {self.score_name}' )
        # ax2.set_ylim(0.25, 1.)
        ax2.set_title(self.score_name)
        ax2.legend()

        fig_name= f'loss_score_plot{VERSION}.png'
        fig_path = f"result/{fig_name}"
        print(f'Saving loss/acc curves at {fig_path}')
        f.savefig(fig_path)
        print(f'Exporting figure to GC storage')
        save_image_to_gcp(fig_name)

    def savemodel(self):
        file_name = f'model_{VERSION}.h5'
        self.model.save(file_name)
        # joblib.dump(self.model, joblib_name)
        save_model_to_gcp(file_name)

    def train(self):
        # step 3 : train
        print('fit')
        self.fit_model()

        # step 4 : evaluate perf
        print('evaluate')
        self.evaluate_model()

        # step 5 : save training loss score
        print('save fig')
        #self.save_fig()

        print('save classification')
        #self.get_classification_report()

        # step 6 : save the trained model
        print('save model')
        self.savemodel()

        print(f'End of {self.name}!')

if __name__ == "__main__":

    VERSION = f'V1_DPoint_NCLass_{NUM_CLASSES}'

    X, y, class_names = create_df(CLASSES)

    X_train, y_train, X_test, y_test = create_train_test_val(X,y)
    save_df_to_gcp(X_test, y_test, y_train, class_names)

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    X_train = reshape_X(X_train)
    X_test = reshape_X(X_test)

    trainer = Trainer(X_train=X_train ,y_train=y_train,
                      X_test=X_test, y_test=y_test)

    trainer.train()