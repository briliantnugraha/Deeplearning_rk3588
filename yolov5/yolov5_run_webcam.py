from yolov5_run import Yolov5, draw_output

# ref vidcapture: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
# import the opencv library
import cv2

model = Yolov5()
modelpath = '../models/yolov5s.rknn'
model._load_model(modelpath)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,  480))

# define a video capture object
vid = cv2.VideoCapture(0)

ctr = 0
start = False
while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	print('frame: ', frame.shape)

	# Display the resulting frame
	# cv2.imshow('frame', frame)
    
	if ctr > 100:
		print(ctr, 'write')
		boxes, classes, scores = model(frame)
		if boxes is not None:
			frame = draw_output(frame, boxes, classes, scores)
		out.write(frame)
	key = cv2.waitKey(1)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
    
	if ctr == 10000:
		print('break True')
		break
	ctr += 1

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
