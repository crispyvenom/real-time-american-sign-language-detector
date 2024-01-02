import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

st.set_page_config(
    page_title="Sign Language Converter"
)
st.sidebar.success("Select a page above.")


class HandTrackingTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                       "T", "U", "V", "W", "X", "Y", "Z"]

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            imgOutput = img.copy()

            hands, img = self.detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)
                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                              (x - self.offset + 90, y - self.offset - 50 + 50), (255, 0, 255), cv2.FILLED)

                if 0 <= index < len(self.labels):
                    cv2.putText(imgOutput, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                (255, 255, 255), 2)
                else:
                    print("Invalid index:", index)

                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                              (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)
        except AttributeError as e:
            st.success("Your Result is Here !")

        return imgOutput


def main():
    st.title("Real Time American Sign Language To Speech Converter")

    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-recognition",
        video_transformer_factory=HandTrackingTransformer,
        async_transform=True,
        request_timeout=30,
    )

    if webrtc_ctx.video_transformer:
        st.image(webrtc_ctx.video_transformer, channels="BGR")


if __name__ == "__main__":
    main()
