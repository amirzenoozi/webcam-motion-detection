import cv2
import numpy as np

def main():
    frame_count = 0
    danger_frame = 0
    previous_frame = None
    vidcap = cv2.VideoCapture(0)
    cv_font = cv2.FONT_HERSHEY_SIMPLEX
            
    # Set Default Value For Text & Color
    text = 'SAFE'
    color = (0, 255, 0)
    
    # Set a smaller resolution
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check Webcam is Exist or Not
    if vidcap.isOpened():
        while (True):
            # blocks until the entire frame is read
            success, img = vidcap.read()
            frame_count += 1

            # 1. Load image; convert to RGB
            img_brg = np.array(img)
            img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)

            # 3. Set previous frame and continue if there is None
            if (previous_frame is None):
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

            # 6. Find The Contours Boxes From Frame
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # 7. Segment Changes
            if len(contours) != 0:
                danger_frame = frame_count
                c = max(contours, key = cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(img=img_brg, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            
                # # Draw All Boxes
                # for contour in contours:
                #     if cv2.contourArea(contour) < 200:
                #         # too small: skip!
                #         continue
                #     (x, y, w, h) = cv2.boundingRect(contour)
                #     cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                    
                # get boundary of this text
                text = 'DANGER'
                color = (0, 0, 255)
            
            elif frame_count - danger_frame > 30:
                text = 'SAFE'
                color = (0, 255, 0)
                
            textsize = cv2.getTextSize(text, cv_font, 1, 2)[0]
            textX = round((640 - textsize[0]) / 2)
            textY = round(textsize[1] + 10)
            cv2.putText(img_brg, text, (textX, textY), cv_font, 1.0, color, 2, cv2.LINE_AA)


            # 8. display image
            cv2.imshow("webcam", img_brg)

            # 9. wait 1ms for ESC to be pressed
            key = cv2.waitKey(1)
            if (key == 27):
                break

        # release resources
        cv2.destroyAllWindows()
        vidcap.release()

if __name__ == '__main__':
    main()