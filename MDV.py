import cv2
import pandas
import datetime
import matplotlib.pyplot as plt


first_frame = None
status_list = [None, None]
times = []

# DataFrame to store time values during which object detection and movement appears
df = pandas.DataFrame(columns=["Start", "End", "Duration"])

# reading the first frame/image of the video
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

no_of_frames = 1
while True:
    no_of_frames = no_of_frames + 1
    check, frame = video.read()
    status = 0

    # converting each frame into gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converting each frame into gaussian blur image
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    print(frame)

    # storing the first frame/image of the video
    if first_frame is None:
        first_frame = gray
        continue

    # calculating the difference between first frame and other frames
    delta_frame = cv2.absdiff(first_frame, gray)

    th_delta = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    th_delta = cv2.dilate(th_delta, None, iterations=0)
    (cnts, _) = cv2.findContours(
        th_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for contour in cnts:
        if cv2.contourArea(contour) < 2000:
            continue
        status = 1
        # creating a rectangular box around the object in the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # List of status for every frame
    status_list.append(status)

    # Recording datetime in a list when change occurs
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.datetime.now())

    cv2.imshow("Frame", frame)
    cv2.imshow("Capturing", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("thresh", th_delta)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# print(status_list)
# print(times)

# Storing time values in DataFrame
for i in range(0, len(times) - 1, 2):
    df = df.append(
        {"Start": times[i], "End": times[i + 1], "Duration": (times[i + 1] - times[i])},
        ignore_index=True,
    )

# Writing DataFrame to a csv file
df.to_csv("Motion_Times" + ".csv")
video.release()
cv2.destroyAllWindows()


plt_1 = plt.figure(figsize=(10, 10))
plt.style.use("bmh")
df = pandas.read_csv("Motion_Times.csv")
x = []
y = []
for i in range(len(df["Duration"])):
    time = df["Duration"][i].split()[2].split(".")[0]
    date_time = datetime.datetime.strptime(time, "%H:%M:%S")
    a_timedelta = date_time - datetime.datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    x.append(df["Start"][i].split(".")[0])
    y.append(seconds)


plt.xlabel("Starting Time", fontsize=20)
plt.ylabel("Duration", fontsize=20)
plt.bar(x, y, width=0.4)
plt.show()
