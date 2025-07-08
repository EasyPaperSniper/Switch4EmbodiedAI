import cv2

def main():
    # List all available video capture devices
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1

    print("Available camera indexes:", arr)

    # Replace '0' with your capture card index if needed
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Cannot open capture card")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('Nintendo Switch Stream', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()