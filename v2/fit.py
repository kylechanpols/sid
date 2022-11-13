import datetime
import os

main_path = "F:/gis/sidv2/"
checkpoint_dir = main_path+"/weights"
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

exec(open(os.path.join(main_path + "/model_compile.py")).read())

log_dir = "logs/fit/" + run_id

# Tensorboard logging callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# checkpoint callback
checkpoint_path = main_path+"weights/"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# On GPU
model_history = model.fit(dataset['train'], epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=dataset['dev'],
                          callbacks = [tensorboard_callback, cp_callback])




