# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import find_finger as ff


args = {
    "model": "./model/export_model_008/frozen_inference_graph.pb",
    # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_015/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    # "labels":"record/classes.pbtxt" ,
    "num_classes": 1,
    "min_confidence": 0.6,
    "class_model": "../model/class_model/p_class_model_1552620432_.h5"}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

if __name__ == '__main__':
    model = tf.Graph()

    with model.as_default():
        print("> ====== loading NAIL frozen graph into memory")
        graphDef = tf.GraphDef()

        with tf.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        # sess = tf.Session(graph=graphDef)
        print(">  ====== NAIL Inference graph loaded.")
        # return graphDef, sess


    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            drawboxes = []
            # cap = cv2.VideoCapture(url)
            vs = WebcamVideoStream(src=1)
            vs.start()
            while True:
                frame = vs.read()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                image = frame
                (H, W) = image.shape[:2]
                # print("H,W:", (H, W))
                output = image.copy()
                img_ff, bin_mask, res = ff.find_hand_old(image.copy())
                image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                (boxes, scores, labels, N) = sess.run(
                    [boxesTensor, scoresTensor, classesTensor, numDetections],
                    feed_dict={imageTensor: image})
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)
                boxnum = 0
                box_mid = (0, 0)
                # print("scores_shape:", scores.shape)
                for (box, score, label) in zip(boxes, scores, labels):
                    # print(int(label))
                    # if int(label) != 1:
                    #     continue
                    if score < args["min_confidence"]:
                        continue
                    # scale the bounding box from the range [0, 1] to [W, H]
                    boxnum = boxnum + 1
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    endY = int(endY * H)
                    X_mid = startX + int(abs(endX - startX) / 2)
                    Y_mid = startY + int(abs(endY - startY) / 2)
                    box_mid = (X_mid, Y_mid)
                    # draw the prediction on the output image
                    label_name = 'nail'
                    # idx = int(label["id"]) - 1
                    idx = 0
                    label = "{}: {:.2f}".format(label_name, score)
                    cv2.rectangle(output, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(output, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
                # show the output image
                # print(boxnum)
                if box_mid == (0, 0):
                    drawboxes.clear()
                    cv2.putText(output, 'Nothing', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                elif boxnum == 1:
                    drawboxes.append(box_mid)
                    if len(drawboxes) == 1:
                        pp = drawboxes[0]
                        cv2.circle(output, pp, 0, (0, 0, 0), thickness=3)
                        # cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                    if len(drawboxes) > 1:
                        num_p = len(drawboxes)
                        for i in range(1, num_p):
                            pt1 = drawboxes[i - 1]
                            pt2 = drawboxes[i]
                            # cv2.circle(output, pp, 0, (0, 0, 0), thickness=3)
                            cv2.line(output, pt1, pt2, (0, 0, 0), 2, 2)
                    cv2.putText(output, 'Point', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                else:
                    drawboxes.clear()
                    cv2.putText(output, 'Nothing', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow("Output", output)
                # cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

                    # vs.stop()
