import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgAr = cv2.imread('Ar1.jpg')
videoAr = cv2.VideoCapture('aslan.mp4')

detection = False
frameCounter = 0
succes, imgVideo = videoAr.read()
hT, wT, cT = imgAr.shape
imgVideo = cv2.resize(imgVideo,(wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
k1, des1 = orb.detectAndCompute(imgAr, None)
#imgAr = cv2.drawKeypoints(imgAr, k1, None)

while True:
    succes, frame1 = cap.read()
    imgAug = frame1.copy()
    k2, des2 = orb.detectAndCompute(frame1, None)
    #frame1 = cv2.drawKeypoints(frame1, k2, None)

    if detection == False:
        videoAr.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == videoAr.get(cv2.CAP_PROP_FRAME_COUNT):
            videoAr.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        succes, imgVideo = videoAr.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    print(len(good))
    imageFeatures = cv2.drawMatches(imgAr, k1, frame1, k2, good, None, flags=2)

    if len(good) > 20:
        detection = True
        srcPts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0], [0,hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(frame1, [np.int32(dst)], True, (255,0,255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (frame1.shape[1], frame1.shape[0]))

        maskNew = np.zeros((frame1.shape[0], frame1.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)

        imgAug = cv2.bitwise_and(imgAug, imgAug, mask= maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

    #cv2.imshow("ImgWarp", maskInv)
    cv2.imshow("ImgWarp", imgAug)
    #cv2.imshow("kamera", frame1)
    #cv2.imshow("Img2", img2)
    #cv2.imshow("Target Image", imgAr)
    #cv2.imshow("Hedef", imageFeatures)
    frameCounter +=1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
