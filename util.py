import torch
import cv2
import numpy as np

def get_optical_flow_input(RGBsequence):
    """
    Get the optical flow information from frame sequence(RGB).

    """

    temp = RGBsequence
    Flow = torch.zeros(temp.size(0),temp.size(1)-1,temp.size(2)-1,temp.size(3),temp.size(4))
    temp = (temp * 0.5 + 0.5) * 255

    frame_prev = temp[0,:,0,:,:].cpu().numpy().transpose((1,2,0)).astype(np.uint8)
    hsv = np.zeros_like(frame_prev)
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    hsv[...,1] = 255

    for b in range(RGBsequence.size(0)):
        for d in range(RGBsequence.size(2)):
            if d < RGBsequence.size(2)-1:

                frame_next = temp[b,:,d+1,:,:].cpu().numpy().transpose((1,2,0)).astype(np.uint8)
                frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(frame_prev, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = flow.transpose((2,0,1))
                flow -= np.mean(flow)

                Flow[b,:,d,:,:] = torch.from_numpy(flow).float()

                frame_prev = frame_next

            else:
                break

    return Flow.cuda()
