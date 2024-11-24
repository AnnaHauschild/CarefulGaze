"""The New Demo with html text and links and output many info and it has many customizable features"""

import gdown
import gradio as gr
import yarp
import sys
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
import csv
import pickle

from ai.detection import detect
from interaction_per_frame_uncertainty import LAEO_computation
from utils.hpe import hpe, project_ypr_in2d
from utils.img_util import resize_preserving_ar, percentage_to_pixel, draw_key_points_pose, \
    visualize_vector, draw_axis, draw_axis_3d, draw_cones, draw_detections 

def yarpdebug(msg):
    print("[DEBUG]" + msg)

def yarpinfo(msg):
    print("[INFO] {}".format(msg))

def yarperror(msg):
    print("\033[91m[ERROR] {}\033[00m".format(msg))

def visualise_hpe(head_Pose_representation,yaw, pitch, roll, image=None, tdx=None, tdy=None, size=50, yaw_uncertainty=-1, pitch_uncertainty=-1, roll_uncertainty=-1, openpose=False, title="", color=(255, 0, 0)):
    if str(head_Pose_representation).lower() == 'vector':
        vector = project_ypr_in2d(yaw, pitch,roll)
        image = visualize_vector(image, [tdx, tdy], vector, title=title, color=color, thickness_lines=1)
        return image
    elif str(head_Pose_representation).lower() == 'axis':
        image = draw_axis_3d(yaw, pitch, roll, image=image, tdx=tdx, tdy=tdy, size=size, thickness_lines=1)
        return image
    elif str(head_Pose_representation).lower() == 'cone':
        _, image = draw_cones(yaw, pitch, roll, unc_yaw=yaw_uncertainty, unc_pitch=pitch_uncertainty, unc_roll=roll_uncertainty, image=image, tdx=tdx, tdy=tdy, size=size)
        return image
    else:
        return image

class headPoseModule(yarp.RFModule):
    """
    Description:
        Class to recognize head pose from iCub cameras
    Args:
        input_port  : images from cameras
    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Input port
        self.image_in_port = yarp.BufferedPortImageRgb()

        # Output port
        self.output_img_port = yarp.BufferedPortImageRgb()
        self.display_buf_image = yarp.ImageRgb()
        self.display_buf_array = None
        self.annotations_port = yarp.BufferedPortBottle()
        self.head_pose_port = yarp.BufferedPortBottle()
  
        # Create numpy array to receive the image and the YARP image wrapped around it
        self.input_img_array = None
        self.width_img = 640 #default, the size will be updated automatically
        self.height_img = 480 #default, the size will be updated automatically

        # frame counter for csv file
        self.frame_count = 0
        # state for csv file
        self.state = "initialize"

    def configure(self, rf):

        if rf.check('help') or rf.check('h'):
            print("Head Pose options:")
            print("\t--name (default headPoseEstimation) module name")
            print("\t--width (default 640) width of the input image")
            print("\t--height (default 480) height of the input image")
            print("\t--threshold (default 0.45) threshold for the detection")
            print("\t--pose_representation (default Vector, alternatives Axis, None) which representation to show")
            print("\t--help print this help")
            return False

        self.process = rf.check('process', yarp.Value(True), 'enable automatic run').asBool()

        self.module_name = rf.check("name",
                                    yarp.Value("headPoseEstimation"),
                                    "module name (string)").asString()
        self.path = rf.check("path",
                             yarp.Value(
                             "../../LAEO_demo_data/examples")).asString()
        self.gaze_model_path = rf.check("gaze_model_path",
                             yarp.Value(
                             "../../LAEO_demo_data/head_pose_estimation")).asString()
        self.model_path = rf.check("model_path",
                             yarp.Value(
                             "../../LAEO_demo_data/keypoint_detector/centernet_hg104_512x512_kpts_coco17_tpu-32")).asString()
        
        if not os.path.exists(self.model_path):
            yarperror("model_path does not exist")
            return False

        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Open ports
        self.image_in_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.output_img_port.open('/' + self.module_name + '/image:o')
        self.annotations_port.open('/' + self.module_name + '/annotations:o')   
        self.head_pose_port.open('/' + self.module_name + '/head_pose:i')

        # format of the output image
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.display_buf_image.setExternal(self.display_buf_array, self.width_img, self.height_img)

        # Initialize variables
        self.width_img = rf.check('width', yarp.Value(640),
                                  'Width of the input image').asInt32()
        self.height_img = rf.check('height', yarp.Value(480),
                                   'Height of the input image').asInt32()
        self.detection_threshold = rf.check('threshold', yarp.Value(0.45),
                                   'Threshold for the detection').asFloat32()
        self.head_Pose_representation = rf.check('pose_representation', yarp.Value("Vector"),
                                   'Which representation to show').asString() #"Vector", "Axis", "None"
        self.input_shape_od_model = (512, 512)
        self.min_score_thresh, self.max_boxes_to_draw, self.min_distance = .25, 50, 1.5
        self.size_plots=50

        # Open csv file to safe data
        self.csv_file = open('head_pose.csv', 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame", "Yaw", "Pitch", "Roll", "X", "Y", "TestPerson1"])

        self.careful_model_path = '/usr/local/robot/cognitiveInteraction/head-pose-estimation/random_forest_model.pkl'
        self.careful_scaler_path = '/usr/local/robot/cognitiveInteraction/head-pose-estimation/standard_scaler.pkl'
        self.input_scaled = []
        self.sequence_length = 10
        self.prev_yaw = 0
        self.prev_pitch = 0
        self.prev_roll = 0
        self.prev_tdx = 0
        self.prev_tdy = 0
        self.input_sequence = np.array([]) #added 
        
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        try:
            self.gaze_model = tf.keras.models.load_model(self.gaze_model_path, custom_objects={"tf": tf})
            yarpinfo("Gaze model successfully loaded")
        except Exception as e:
            yarperror(f"Cannot load gaze_model please check the path {self.gaze_model_path}")
            yarperror(e)
            return False 

        try:
            # Load the model from the file
            with open(self.careful_model_path, 'rb') as file:
                self.careful_model = pickle.load(file)

            yarpinfo("Careful model successfully loaded")

        except Exception as e:
            yarperror(f"Cannot load careful_model please check the path {self.careful_model_path}")
            yarperror(e)
            return False   

        try:
            # Load the scaler from the file
            with open(self.careful_scaler_path, 'rb') as file:
                self.careful_scaler = pickle.load(file)

            yarpinfo("Careful Scaler successfully loaded")

        except Exception as e:
            yarperror(f"Cannot load careful_scaler please check the path {self.careful_scaler_path}")
            yarperror(e)
            return False      
        
        try:
            self.model = tf.saved_model.load(os.path.join(self.model_path, 'saved_model'))
            yarpinfo("Model successfully loaded")
        except Exception as e:
            yarperror(f"Cannot load model please check the path {self.model_path}")
            print(e)
            return False      

        self.DEBUG = False
        if self.DEBUG:
            if yarp.Network.connect("/image:o", "/"+self.module_name+"/image:i"): #webcam/image:o
                yarpdebug("LOCAL DEBUG: Connected to webcam")

        yarpinfo('Module initialization done, running')

        return True
    
    def carefull_prediction(self, yaw=0, pitch=0, roll=0, tdx=0, tdy=0):
        delta_yaw = abs(yaw - self.prev_yaw)
        delta_pitch = abs(pitch - self.prev_pitch)
        delta_roll = abs(roll - self.prev_roll)
        delta_tdx = abs(tdx - self.prev_tdx)
        delta_tdy = abs(tdy - self.prev_tdy)
        input = np.array([delta_yaw, delta_pitch, delta_roll, delta_tdx, delta_tdy])
        input_scaled = self.careful_scaler.transform(input)
        self.input_sequence.extend(input_scaled)
        if len(self.input_sequence) > self.sequence_length*len(input_scaled):
            self.input_sequence = self.input_sequence[-self.sequence_length*len(input_scaled):]

            statistical_features = ['min', 'max', 'mean', 'std']
            axis_features = ['Yaw', 'Pitch']  
            for feature in statistical_features:
                for axis in axis_features:
                    relevant_cols = [col for col in self.input_sequences.columns if 'delta_' + axis in col]
                    self.input_sequences[f'{axis}_{feature}'] = self.input_sequences[relevant_cols].agg(feature, axis=1)
                    self.input_sequences.reshape(1, -1)
            
            carefulness = self.careful_model.predict(self.input_sequence)
        else:
            carefulness = 1 # When the model does not hve enough data to build the sequence, the function is conservative and outputs full

        return carefulness


    def interruptModule(self):
        yarpinfo("stopping the module")
        self.handle_port.interrupt()
        self.image_in_port.interrupt()
        self.output_img_port.interrupt()

        return True
    
    def getPeriod(self):
        return 0.001

    def close(self):
        self.handle_port.close()
        self.image_in_port.close()
        self.output_img_port.close()
        self.csv_file.close()
        self.head_pose_port.close()

        return True
       
    def get_image_from_bottle(self, yarp_image):
        """
        Format a yarp image to openCV Mat
        :param yarp_img:
        :return: openCV::Mat
        """
        # Check image size, if differend adapt the size (this check takes 0.005 ms, we can afford it)
        if yarp_image.width() != self.width_img or yarp_image.height() != self.height_img:
            yarpdebug("imput image has different size from default 640x480, fallback to automatic size detection")
            self.width_img = yarp_image.width()
            self.height_img = yarp_image.height()
            self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

        # Convert yarp image to numpy array
        yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
        frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3))
        
        #convert to rgb
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # covert at grey scale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # it is still grey scale but with 3 channels to add the colours of the points and lines

        return frame
    
    def write_annotated_image(self,annotated_image):
        """
        Stream the annotated image on a yarp port
        """

        #convert annotated image to rgb format
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        # Stream annotated image
        self.display_buf_image = self.output_img_port.prepare()
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(annotated_image.tobytes(), self.width_img, self.height_img)
        self.output_img_port.write()

    #Send pose data to the state machine
    def send_pose_data(self, values):
        bottle = self.head_pose_port.prepare()
        bottle.clear()
        for value in values:
            bottle.addFloat64(value)
        self.head_pose_port.write()

    '''def read_commands(self):
        command = self.head_pose_port.read(False)  # Non-blocking read
        if command is not None:
            message = command.toString()
            return message
        return None'''
    def read_commands(self):
        command = self.head_pose_port.read(False)  # Non-blocking read
        if command is not None:
            message = command.get(0).asString()  # Erster String
            trial_id = command.get(1).asString()  # Zweiter String
            combined_message = f"{message} {trial_id}"  # Kombinieren der Strings
            return combined_message
        return None

    def updateModule(self):
        yarpinfo("update module")
        """input_yarp_image = self.image_in_port.read(False)
        # read yarp image
        if input_yarp_image is not None:
            frame_input = self.format_yarp_image(input_yarp_image)
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()"""
        if self.process:

            # Read image from port
            message = self.image_in_port.read(False)

            if message is not None:

                # Read yarp image                
                img = self.get_image_from_bottle(message)
                img_resized, new_old_shape = resize_preserving_ar(img, self.input_shape_od_model)

                print('inference centernet')
                detections, elapsed_time = detect(self.model, img_resized, self.detection_threshold,
                                                new_old_shape)  # detection classes boxes scores
                #print(detections)

                img_with_detections = draw_detections(img_resized, detections, self.max_boxes_to_draw, None, None, None)
                # cv2.imshow("aa", img_with_detections)

                det, kpt = percentage_to_pixel(img.shape, detections['detection_boxes'], detections['detection_scores'],
                                        detections['detection_keypoints'], detections['detection_keypoint_scores'])
                
                people_list = []

                print('inferece hpe')

                for j, kpt_person in enumerate(kpt):
                    # kpt person 
                    yaw, pitch, roll, tdx, tdy = hpe(self.gaze_model, kpt_person, detector='centernet')
                    print(f'yaw {yaw}, pitch {pitch}, roll {roll}')

                
                    #TODO: DONE: save to dump file
                    yaw_value, yaw_uncertainty = yaw.numpy().tolist()[0]
                    pitch_value, pitch_uncertainty = pitch.numpy().tolist()[0]
                    roll_value, roll_uncertainty = roll.numpy().tolist()[0] #test
                    #self.csv_writer.writerow([self.frame_count, yaw_value, pitch_value, roll_value, tdx, tdy])
                    carefulness = self.carefull_prediction(yaw_value, pitch_value, roll_value, tdx, tdy)
                    print(f"Carefulness: {'Full' if carefulness else 'Empty'}")



                    #TODO: DONE: send pose data to the state machine
                    pose_values = [yaw_value, pitch_value, roll_value]
                    #self.send_pose_data(pose_values) 
                    #self.state = self.head_pose_port.read()  #TODO: change name to message state machine
                    #if (self.state == "record"):
                    command = self.read_commands()
                    if command:
                        #self.last_command = command  # store message
                        #command = command
                        #print(f"Received message: {self.last_command}")
                        print(f"Received message: {command}")
                        self.csv_writer.writerow([command])
                    self.csv_writer.writerow([self.frame_count, yaw_value, pitch_value, roll_value, tdx, tdy])

                    # img = draw_axis_3d(yaw[0].numpy()[0], pitch[0].numpy()[0], roll[0].numpy()[0], image=img, tdx=tdx, tdy=tdy,
                    #                    size=50)

                    people_list.append({'yaw'      : yaw[0].numpy()[0],
                                        'yaw_u'    : 0,
                                        'pitch'    : pitch[0].numpy()[0],
                                        'pitch_u'  : 0,
                                        'roll'     : roll[0].numpy()[0],
                                        'roll_u'   : 0,
                                        'center_xy': [tdx, tdy]
                                        })
                self.frame_count += 1
                for i in range(len(det)):
                    img = draw_key_points_pose(img, kpt[i])
                
                # call LAEO
                clip_uncertainty = 0.5
                binarize_uncertainty = False
                interaction_matrix = LAEO_computation(people_list, clipping_value=clip_uncertainty,
                                                clip=binarize_uncertainty)
                print(f'Head pose representation: {self.head_Pose_representation}')

                for index, person in enumerate(people_list):
                    green = round((max(interaction_matrix[index, :])) * 255)
                    colour = (0, green, 0)
                    if green < 40:
                        colour = (0, 0, 255)
                        
                    img = visualise_hpe(self.head_Pose_representation, person['yaw'], person['pitch'], person['roll'], image=img, tdx=person['center_xy'][0], tdy=person['center_xy'][1], size=self.size_plots, yaw_uncertainty=person['yaw_u'], pitch_uncertainty=person['pitch_u'], roll_uncertainty=person['roll_u'], title="", color=colour)
        
                    #vector = project_ypr_in2d(person['yaw'], person['pitch'], person['roll'])
                    #img = visualize_vector(img, person['center_xy'], vector, title="", color=colour)
                uncertainty_mean = [i['yaw_u'] + i['pitch_u'] + i['roll_u'] for i in people_list]
                uncertainty_mean_str =  ''.join([str(round(i, 2)) + ' ' for i in uncertainty_mean])
                yarpinfo(f"Uncertainty mean: {uncertainty_mean_str}")
                # DEBUG: Show annotated image without streaming it
                if self.DEBUG and img is not None:
                    cv2.imshow('MaLGa Lab Demo', img)

                # Write annontated image if connected
                if self.output_img_port.getOutputCount():
                    self.write_annotated_image(img)
                    cv2.waitKey(1) & 0xff

                # Write annotations on the port if connected
                if self.annotations_port.getOutputCount():
                    #create bottle
                    bottle = self.annotations_port.prepare()


                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    return
                # Update the flag and frame as last step of the loop
                #img = self.get_image_from_bottle(message)
                frame = None

            # Destroy all the windows
            cv2.destroyAllWindows()

        return True

if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    headPose = headPoseModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('headPoseEstimation')
    rf.setDefaultConfigFile('headPoseEstimation.ini')

    if rf.configure(sys.argv):
        headPose.runModule(rf)
    sys.exit()





