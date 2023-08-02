# -*- coding: utf-8 -*-

# import the necessary packages
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import find_finger as ff

tf.disable_v2_behavior()

args = {
    "model": "./model/export_model_008/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    "num_classes": 1,
    "min_confidence": 0.5,
    "class_model": "../model/class_model/p_class_model_1552620432_.h5"
}

# COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))
COLORS = [(255, 0, 0)] # Blue in BGR format


def run_inference(sess, image, boxesTensor, scoresTensor, classesTensor, numDetections):
    (boxes, scores, labels, N) = sess.run(
        [boxesTensor, scoresTensor, classesTensor, numDetections],
        feed_dict={imageTensor: image})

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

def process_results(boxes, scores, labels, H, W):
    boxnum = 0
    box_mid = (0, 0)
    valid_boxes = [(box, score, label) for (box, score, label) in zip(boxes, scores, labels) if score >= args["min_confidence"]]
    for (box, score, label) in valid_boxes:
        boxnum += 1
        (startY, startX, endY, endX) = box
        startX = int(startX * W)
        startY = int(startY * H)
        endX = int(endX * W)
        endY = int(endY * H)
        X_mid = startX + int(abs(endX - startX) / 2)
        Y_mid = startY + int(abs(endY - startY) / 2)
        box_mid = (X_mid, Y_mid)

        label_name = 'nail'
        idx = 0
        label = "{}: {:.2f}".format(label_name, score)
        cv2.rectangle(output, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

    return boxnum, box_mid

if __name__ == '__main__':
    model = tf.Graph()

    with model.as_default():
        print("> ====== loading NAIL frozen graph into memory")
        graphDef = tf.GraphDef()

        with tf.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        print(">  ====== NAIL Inference graph loaded.")

    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            drawboxes = []
            vs = cv2.VideoCapture(0)

            while True:
                _, frame = vs.read()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                image = frame
                (H, W) = image.shape[:2]
                output = image.copy()
                img_ff, bin_mask, res = ff.find_hand_old(image.copy())
                image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                boxes, scores, labels = run_inference(sess, image, boxesTensor, scoresTensor, classesTensor, numDetections)
                boxnum, box_mid = process_results(boxes, scores, labels, H, W)
                
                # boxnum = numero de unhas detectadas
                # box_mid = posicao da unha na imagem
                # boxes = posicao de cada unha, array de 4, n
                # labes = nao sei
                # scores = acuracia de cada unha detectada

                # if box_mid == (0, 0):
                #     drawboxes.clear()
                #     cv2.putText(output, 'Nothing', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                # elif boxnum == 1:
                #     drawboxes.append(box_mid)
                #     if len(drawboxes) == 1:
                #         pp = drawboxes[0]
                #         cv2.circle(output, pp, 0, (0, 0, 0), thickness=3)
                #     if len(drawboxes) > 1:
                #         num_p = len(drawboxes)
                #         for i in range(1, num_p):
                #             pt1 = drawboxes[i - 1]
                #             pt2 = drawboxes[i]
                #             cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                #     cv2.putText(output, 'Point', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                # else:
                #     drawboxes.clear()
                #     # cv2.putText(output, 'Nothing', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow("Output", output)
                
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
