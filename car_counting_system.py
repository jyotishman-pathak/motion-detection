import numpy as np
import cv2
VIDEO_SOURCE = 'videos/Cars.mp4'
VIDEO_OUT = 'videos/results/car_counting_system.avi'
cap = cv2.VideoCapture(VIDEO_SOURCE)
has_frame , frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 30.0, (frame.shape[1], frame.shape[0]),False)


frames_id = cap.get(cv2.CAP_PROP_FRAME_COUNT)* np.random.uniform(size = 30)


frames = []
for fid in frames_id:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame , frame = cap.read()
    frames.append(frame)


median_frame = np.median(frames, axis = 0).astype(dtype=np.uint8)
cv2.imwrite('model_median_frame.jpg', median_frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
gray_median_frame = cv2.cvtColor(median_frame , cv2.COLOR_BGR2GRAY)

car_count = 0
centroids_prev = []
while True:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame_gray, gray_median_frame)
    _, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Remove small blobs (like people inside cars) by morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dframe = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
    dframe = cv2.morphologyEx(dframe, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            current_centroids.append((cx, cy))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #  Defining the  horizontal line
    line_y = frame.shape[0] // 2

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

    for cx, cy in current_centroids:
        if line_y - 6 < cy < line_y + 6:

            near_prev = any(abs(cx - pcx) < 25 and abs(cy - pcy) < 25 for pcx, pcy in centroids_prev)
            if not near_prev:
                car_count += 1
                print("Car count:", car_count)

    centroids_prev = current_centroids

    cv2.putText(frame, f"Cars: {car_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.imshow('Count', frame)
    cv2.imshow('Movement Mask', dframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()