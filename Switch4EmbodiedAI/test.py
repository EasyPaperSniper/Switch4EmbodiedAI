import cv2

def main(capture_card_index=0):
    cap = cv2.VideoCapture(capture_card_index)  

    if not cap.isOpened():
        print("Cannot open capture card")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        

        # process the frame if needed, such as masking, resizing, etc.
        cv2.imshow('Nintendo Switch Stream', frame)







        if cv2.waitKey(1)  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    capture_card_index = 0  # Set your capture card index here
    main(capture_card_index)