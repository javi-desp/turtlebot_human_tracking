import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist



# Load the trained network "MobileNetSSD"
global network 
network = cv2.dnn.readNetFromCaffe('../trained_network/MobileNetSSD_deploy.prototxt.txt', '../trained_network/MobileNetSSD_deploy.caffemodel')

class CVControl:
	def __init__(self):

		# Define the velocity topic of turtlebot3
		self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
		self.cmd = Twist()

		# Define the image topic of turtlebot3
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.img_callback)

	# Main function
	def img_callback(self, data):
		person_detected = 0
		area_biggest_detection = 0
		center_biggest_detection = 0

		# Proportional control variables
		K_rotation = 0.002
		K_velocity = 0.0000045
		max_velocity = 0.25 

		# Initialize the list of objects labels that the trained network can detect and define box of detections
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
		
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print (e)
		
		# Preprocessing of the frame
		cv_gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
		frame = cv_image
		frame_resized = cv2.resize(frame, (300, 300))
		
		# Get high and width of the real frame
		(h, w) = frame.shape[:2]

		# Get a blob which is our input image after mean substraction, normalizing and channel swapping
		blob = cv2.dnn.blobFromImage(frame_resized,0.007843, (300, 300), 127.5)

		# Pass the blob through the network and obtain the detections and predictions
		network.setInput(blob)
		detections = network.forward()


		# Loop over all the detections in a same frame
		for i in np.arange(0, detections.shape[2]):
			object_type = detections[0,0,i,1]
			confidence = detections[0, 0, i, 2]
			if object_type == 15 and confidence > 0.2: # If has detected a human and it has a confidence > 20% --> HUMAN DETECTED

				person_detected = 1
				# Get the (x, y) coordinates of the bounding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# Get bounding box center and area used for commanding the TurtleBot to follow a person
				center_box = int((startX+endX)/2)
				area_box = (endX-startX)*(endY-startY)
				
				# Draw the box and the intruder label in the frame 
				label = "{}: {:.2f}%".format('intruder',confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),[0,0,255], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

				# If there are more than 1 human get the position of the closest (biggest area)
				if area_box > area_biggest_detection: 
					area_biggest_detection = area_box
					center_biggest_detection = center_box

		if person_detected:

			# If the detection area has an umbral value
			if area_biggest_detection > 10000: 

				# Proportional controller for the TurtleBot based on the persons position in the frame and how far they are away
				distance_to_the_person = 150000
				v = K_velocity*(distance_to_the_person - area_biggest_detection)

				# Proportional controller for the TurtleBot based on the persons position in the frame and how far they are away
				w = K_rotation*(w/2 - center_biggest_detection)
				
				# Apply max and min to velocity
				v = np.max([-max_velocity, v])
				v = np.min([max_velocity, v])

				# Send movement to the turtlebot
				self.send_command(v, w)	


		cv2.imshow("Image window", frame)
		cv2.waitKey(3)


	# Put velocity (v) and angular velocity (w) to a Twist message and publish it to the turtlebot topic
	def send_command(self, v, w):		
		self.cmd.linear.x = v
		self.cmd.angular.z = w

		self.cmd_pub.publish(self.cmd)


def main():
	ctrl = CVControl()
	rospy.init_node('human_tracking')
	try:
		rospy.spin()
		
	except KeyboardInterrupt:
		print ("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
