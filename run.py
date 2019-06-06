import cv2
import time
import copy
from clstm_network_32 import *

# Open the camera
cap = cv2.VideoCapture('C:/Users/wangyifeng/Desktop/Ballet_Classifier/videos/ballet.avi')
#cap = cv2.VideoCapture(0)

# Initialize the network and input
inputs = torch.zeros(1,3,32,112,112)

clstmNet = CNet().cuda()
clstmNet.load_state_dict(torch.load('weights/clstm_weights_4_11.pkl'))
clstmNet.eval()

# Take first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(prev_frame)
hsv[...,1] = 255

# Create the display window
display = np.zeros((int(prev_frame.shape[0]+prev_frame.shape[1]/2),int(2*prev_frame.shape[1]),3), dtype = np.uint8)

# Save the display video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video_save = cv2.VideoWriter('display.avi',fourcc, 20, (display.shape[0],display.shape[1]))

index = 0
while cap.isOpened():
    ret_, frame = cap.read()
    
    if ret_:
        display[:frame.shape[0],:frame.shape[1],:] = frame
        frame_ = copy.deepcopy(frame)
        frame_opfl = copy.deepcopy(frame)
        curt_gray = cv2.cvtColor(frame_opfl, cv2.COLOR_BGR2GRAY)

        if index == 0:
            start = time.time()

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Take continuous 32 frames(without overlapped) to form input
        frame_ = cv2.resize(frame_, (112,112))
        frame_ = (frame_ / 255 - 0.5) / 0.5
        frame_ = frame_.transpose((2,0,1))
        frame_ = torch.from_numpy(frame_)
        inputs[0,:,index,:,:] = frame_
        inputs = inputs.cuda()
        
        temp = inputs[0,0,31,:,:]
        if temp.sum() != 0:
            with torch.no_grad():
                output, features = clstmNet.forward(inputs)
                prediction = clstmNet.predict(output)
                end = time.time()
                print(prediction)
                print('time: %.4fs' % (end - start))

                # Show the feature maps
                k = 0
                for feature in features:
                    k += 1
                    v_feature = feature[0,0,0,:,:]
                    v_feature = v_feature.cpu().numpy()
                    v_feature = np.round((1.0 / (1 + np.exp(-1 * v_feature))) * 255)
                    v_feature = v_feature.astype(np.uint8)
                    v_feature = cv2.resize(v_feature, (int(frame.shape[1]/2),int(frame.shape[1]/2)))
                    v_feature = cv2.cvtColor(v_feature, cv2.COLOR_GRAY2BGR)
                    display[frame.shape[0]:,(k-1)*v_feature.shape[0]:k*v_feature.shape[0],:] = v_feature
                    #cv2.imshow('feature_conv3d_' + str(k) + '_c1', v_feature)
                    #cv2.waitKey(5)

        if index < 31:
            index += 1
        else:
            index = 0
            inputs = torch.zeros(1,3,32,112,112).cuda()

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        display[:frame.shape[0],frame.shape[1]:,:] = rgb

        #cv2.imshow('original video', frame)
        #cv2.imshow('optical flow',rgb)
        cv2.putText(display, 'Original Video', (0,48), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 6)
        cv2.putText(display, 'Optical Flow', (frame.shape[1],48), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 6)
        cv2.putText(display, 'Feature Maps', (0,frame.shape[0]+24), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        
        #video_save.write(display)
        cv2.imshow('diaplay', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = curt_gray

    if frame is None:
        break

cap.release()
#video_save.release()
cv2.destroyAllWindows()
