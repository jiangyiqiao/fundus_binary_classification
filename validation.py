import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework import get_or_create_global_step
import scripts.inception_preprocessing as inception_preprocessing
from scripts.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim

log_dir = 'logs'
log_eval = 'log_test_eval'
dataset_dir = 'tfdata'
image_size=256
num_classes = 2
batch_size = 128
num_epochs = 3
file_pattern = 'fundus_%s_*.tfrecord'
labels_to_name = {0:"Bad",1:"Good"}

items_to_descriptions = {
    'image': 'A 3-channel RGB fundus image of a patient.',
    'label': 'A label that is as such -- 0:abnromal/sick, 1:normal/healthy'
}
#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)
#============== DATASET LOADING ======================
def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='fundus'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later.
    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting
    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))
    print(file_pattern_path)

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    print("tfrecords_to_count",tfrecords_to_count)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    print("num_samples",num_samples)
    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.
    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels


def run():
    #Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        print("tf.logging")
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('validation', dataset_dir)
        print("dataset")
        images, raw_images, labels = load_batch(dataset, batch_size = batch_size, is_training = False)
        print(labels)
        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch
        print("num_batches_per_epoch,num_steps_per_epoch",num_batches_per_epoch,num_steps_per_epoch)

        #Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)
            print("logists")

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        print("finished variables_to_restore")
        saver = tf.train.Saver(variables_to_restore)
        print("finished tf.train.Saver(variables_to_restore)")
        def restore_fn(sess):
            print(checkpoint_file)
            print("saver.restore(sess, checkpoint_file)")
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        print("predictions",predictions)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        print("accuracy, accuracy_update\n")
        metrics_op = tf.group(accuracy_update)
        print(" metrics_op\n")

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        print("global_step\n")
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        print("global_step_op\n")


        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)

            return accuracy_value


        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()
        print("finished Define some scalar quantities to monitor")
        #Get your supervisor
        sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)
        print("finished tf.train.Supervisor")

        #Now we are ready to run in one session
        with sv.managed_session() as sess:
        #with sv.managed_session() as sess:
            print("begin sv.managed_session()")
            print(int(num_steps_per_epoch * num_epochs))
            for step in range(int(num_steps_per_epoch * num_epochs)):
                print(step)
                sess.run(sv.global_step)
                print("print vital information every start of the epoch as always")
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
            


                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)


                #Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
            print("Now we want to visualize the last batch's images just to see what our model has predicted")
            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions = sess.run([raw_images, labels, predictions])
            for i in range(10):
                image, label, prediction = raw_images[i], labels[i], predictions[i]
                prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
                text = 'Prediction: %s \n Ground Truth: %s' %(prediction_name, label_name)
                print(text)

            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')

if __name__ == '__main__':
    run()
