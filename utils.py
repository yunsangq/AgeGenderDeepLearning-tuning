import cv2

FACE_PAD = 50
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')


def sub_image(name, img, x, y, w, h):
    upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
    lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
    roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
    cv2.imwrite(name, roi_color)
    return name


def faceDetector(image_file, basename):
    img = cv2.imread(image_file)
    min_height_dec = 20
    min_width_dec = 20
    min_height_thresh = 50
    min_width_thresh = 50
    min_h = int(max(img.shape[0] / min_height_dec, min_height_thresh))
    min_w = int(max(img.shape[1] / min_width_dec, min_width_thresh))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(min_h, min_w))
    images = []
    for i, (x, y, w, h) in enumerate(faces):
        images.append(sub_image('%s/%s-%d.jpg' % ('./', basename, i + 1), img, x, y, w, h))

    print('%d faces detected' % len(images))

    return images