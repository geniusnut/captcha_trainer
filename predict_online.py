import argparse
import os
import time

import tensorflow as tf
import flask
import numpy as np

from core import NeuralNetwork, RunMode, ModelConfig
from predict_testing import Predict

app = flask.Flask(__name__)

predictor = Predict('gigsgigs')


def recognize_img(img):
    with open(img, "rb") as f:
        b = f.read()

    batch = predictor.get_image_batch(b)
    if not batch:
        return ''
    st = time.time()
    predict_text = predictor.predict_func(
        batch,
        sess,
        dense_decoded_op,
        x_op,
    )
    return predict_text


graph = tf.get_default_graph()
with graph.as_default():
    sess = tf.Session(
        graph=graph,
        config=tf.ConfigProto(
            # allow_soft_placement=True,
            # log_device_placement=True,
            gpu_options=tf.GPUOptions(
                allocator_type='BFC',
                # allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=0.1
            ))
    )

    sess.run(tf.global_variables_initializer())
    # tf.keras.backend.set_session(session=sess)

    model = NeuralNetwork(
        predictor.model_conf,
        RunMode.Predict,
        predictor.model_conf.neu_cnn,
        predictor.model_conf.neu_recurrent
    )
    model.build_graph()
    model.build_train_op()

    saver = tf.train.Saver(var_list=tf.global_variables())

    """从项目中加载最后一次训练的网络参数"""
    saver.restore(sess, tf.train.latest_checkpoint(predictor.model_conf.model_root_path))
    # model.build_graph()
    # _ = tf.import_graph_def(graph_def, name="")

    """定义操作符"""
    dense_decoded_op = sess.graph.get_tensor_by_name("dense_decoded:0")
    x_op = sess.graph.get_tensor_by_name('input:0')
    """固定网络"""
    sess.graph.finalize()


@app.route("/predict", methods=["POST"])
def web_predict():
    with graph.as_default():
        data = {"code": -1}
        print(flask.request.form)
        if flask.request.method == "POST":
            img = flask.request.form.get('img')
            if img is None:
                data['msg'] = 'No base64'
                return flask.jsonify(data)
            yzm = recognize_img(img)
            data['data'] = {"yzm": yzm}
            data['code'] = 0
        return flask.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recognize captcha from image ≥ <.')
    parser.add_argument('--port', type=int, default=8889, help='web mode port')
    args = parser.parse_args()
    app.run(port=args.port)
