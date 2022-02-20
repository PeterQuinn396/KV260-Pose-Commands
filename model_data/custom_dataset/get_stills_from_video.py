import cv2

def get_stills(vid_name: str):
    cap = cv2.VideoCapture(vid_name)

    print(f"Processing {vid_name}")
    if cap.isOpened() ==False:
        print("Error reading video file")

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(f"Got {video_length} frames")

    image_count = 0
    frame_count = 0
    SAMPLE_EVERY = 30
    FPS = 30
    fname_prefix = vid_name.split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # 1 image every second, drop first 5 secs and last 5 secs
        if frame_count % SAMPLE_EVERY == 0 and frame_count > (5 * FPS) and frame_count < (video_length-5*FPS):
            # Write the results back to output location.
            cv2.imwrite(f"{fname_prefix}_{image_count}.jpg", frame)
            image_count += 1

        # If there are no more frames left
        if (frame_count > (video_length - 1)):
            # Log the time again
            # time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print(f"Done extracting frames.\n {image_count} frames extracted")
            # print ("It took %d seconds forconversion." % (time_end-time_start))
            break


if __name__ == '__main__':
    get_stills('data/down/down.mp4')
    get_stills('data/up/up.mp4')
    get_stills('data/left/left.mp4')
    get_stills('data/right/right.mp4')
    get_stills('data/fist/fist.mp4')
    get_stills('data/palm/palm.mp4')

