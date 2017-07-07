import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from week1.CNN.train import model_freezer

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename",
                        default="/media/royd1990/fd0ff253-17a9-49e9-a4bb-0e4529adb2cb/home/royd1990/Documents/deep_learning_tensorFlow/ml-course1/week1/CNN/train/checkpoints/convnet_mnist/frozen_model.pb",
                        type=str,
                        help="Frozen model file to import")
    mnist = input_data.read_data_sets(
        "/media/royd1990/fd0ff253-17a9-49e9-a4bb-0e4529adb2cb/home/royd1990/Documents/deep_learning_tensorFlow/ml-course1/week1/CNN/train/mnist",
        one_hot=True)
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = model_freezer.load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    X = graph.get_tensor_by_name('prefix/data/X_placeholder:0')
#    Y = graph.get_tensor_by_name('prefix/data/Y_placeholder:0')
    preds = graph.get_tensor_by_name('prefix/loss/pred:0')
#    loss = graph.get_tensor_by_name('prefix/loss/loss:0')
    dropout = graph.get_tensor_by_name('prefix/dropout:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        n_batches = int(mnist.test.num_examples / 128)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(128)
            # Note: we didn't initialize/restore anything, everything is stored in the graph_def
          #  _, pred = sess.run([loss, preds], feed_dict={X: X_batch,Y: Y_batch, dropout: 0.75})  # , Y:Y_batch,dropout: 0.75
            # preds = tf.nn.softmax(logits_batch)
           # x = pred[0]
           # correct_preds = tf.equal(tf.argmax(pred,1), tf.argmax(Y_batch, 1))
           # accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
           # total_correct_preds += sess.run(accuracy)
            y_out=sess.run(preds,feed_dict={X: X_batch,dropout: 0.75})
            correct_preds = tf.equal(tf.argmax(y_out, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
        #print(y_out)
        print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))
