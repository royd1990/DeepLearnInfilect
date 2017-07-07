import json, argparse, time

import tensorflow as tf
from week1.CNN.train import model_freezer
from tensorflow.examples.tutorials.mnist import input_data
from flask import Flask, request
from flask_cors import CORS

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']

    ##################################################
    # Tensorflow part
    ##################################################
    n_batches = int(mnist.test.num_examples / 128)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(128)
        y_out = persistent_sess.run(preds, feed_dict={X: X_batch, dropout: 0.75})
        correct_preds = tf.equal(tf.argmax(y_out, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += persistent_sess.run(accuracy)
    ##################################################
    # END Tensorflow part
    ##################################################
    acc=float(float(total_correct_preds)/float(n_batches*128))
    json_data = json.dumps({'acc': acc})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="/media/royd1990/fd0ff253-17a9-49e9-a4bb-0e4529adb2cb/home/royd1990/Documents/deep_learning_tensorFlow/ml-course1/week1/CNN/train/checkpoints/convnet_mnist/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=3.5, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = model_freezer.load_graph(args.frozen_model_filename)
    X = graph.get_tensor_by_name('prefix/data/X_placeholder:0')
    preds = graph.get_tensor_by_name('prefix/loss/pred:0')
    dropout = graph.get_tensor_by_name('prefix/dropout:0')

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    mnist = input_data.read_data_sets(
        "/media/royd1990/fd0ff253-17a9-49e9-a4bb-0e4529adb2cb/home/royd1990/Documents/deep_learning_tensorFlow/ml-course1/week1/CNN/train/mnist",
        one_hot=True)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
app.run()