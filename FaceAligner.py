from mtcnn.mtcnn import MTCNN

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35), #0.2 - 0.8 (0.2 being very zoomed into the face) 
        desiredFaceWidth=256, desiredFaceHeight=None):
        self.detector = MTCNN()
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image): # RGB images
        result = self.detector.detect_faces(image)
        leftEyeCenter = result[0]["keypoints"]["left_eye"]
        rightEyeCenter = result[0]["keypoints"]["right_eye"]
        
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        return output
