import tensorflow.compat.v1 as tf
import numpy as np
import cv2

tf.disable_v2_behavior()

args = {
    "model": "./src/model/nail_inference_graph.pb",
    "min_confidence": 0.5
}

def hex_to_bgr(hex):
    hex = str(hex).replace('#', "")
    red = int(hex[0:2], 16)
    green = int(hex[2:4], 16)
    blue = int(hex[4:6], 16)
    return tuple((blue, green, red))

def hex_list_to_bgr_list(hex_list):
    return [hex_to_bgr(hex_color) for hex_color in hex_list]

# Example usage:
hex_colors = ["#c75685", "#8d435c", "#e393b9"]
COLORS = hex_list_to_bgr_list(hex_colors)
color = 0

def process_image(frame):
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)
    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    return res

def run_inference(sess, image, boxesTensor, scoresTensor, classesTensor, numDetections):
    (boxes, scores, labels, N) = sess.run(
        [boxesTensor, scoresTensor, classesTensor, numDetections],
        feed_dict={imageTensor: image})

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

def process_results(boxes, scores, labels, H, W, idx):
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

        # label_name = 'nail'  
        # label = "{}: {:.2f}".format(label_name, score)

        # Calculate the center and axes lengths of the ellipse
        center = ((startX + endX) // 2, (startY + endY) // 2)
        axes = ((endX - startX) // 2, (endY - startY) // 2)

        # Draw the filled ellipse with the specified color
        cv2.ellipse(output, center, axes, 0, 0, 360, COLORS[idx], -1)

        # Draw the text in white
        # y = startY - 10 if startY - 10 > 10 else startY + 10
        # cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)  
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

            vs = cv2.VideoCapture(0)

            while True:
                _, frame = vs.read()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                (H, W) = frame.shape[:2]
                output = frame.copy()
                res = process_image(frame.copy())

                image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                boxes, scores, labels = run_inference(sess, image, boxesTensor, scoresTensor, classesTensor, numDetections)
                boxnum, box_mid = process_results(boxes, scores, labels, H, W, color)
                
                cv2.imshow("Output", output)

                if cv2.waitKey(1) == ord("a"): color = 0
                if cv2.waitKey(1) == ord("b"): color = 1
                if cv2.waitKey(1) == ord("c"): color = 2

                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break
                
