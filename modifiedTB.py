from keras.callbacks import TensorBoard
import tensorflow as tf
import os, datetime


# Modified tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, log_dir=None,**kwargs):
       # 默认日志目录：logs/20250807-123456
        if log_dir is None:
            log_dir = os.path.join(
                "logs",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        super().__init__(log_dir=log_dir, **kwargs)

        self.step = 1
        # 关键：在 TF-2.x 用 create_file_writer
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overridden, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overridden
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overridden, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        if not stats:
            return

        with self.writer.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.step)
        if self.step % 100 == 0:
            self.writer.flush()
        self.step += 1
