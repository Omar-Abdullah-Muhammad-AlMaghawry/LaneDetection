class FeatureSourcer:
  def __init__(self, p, start_frame):

    self.color_model = p['color_model']
    self.s = p['bounding_box_size']

    self.ori = p['number_of_orientations']
    self.ppc = (p['pixels_per_cell'], p['pixels_per_cell'])
    self.cpb = (p['cells_per_block'], p['cells_per_block'])
    self.do_sqrt = p['do_transform_sqrt']

    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC = None, None, None

    self.RGB_img = start_frame
    self.new_frame(self.RGB_img)

  def hogFn(self, channel):
    features, hog_img = hog(channel,
                            orientations = self.ori,
                            pixels_per_cell = self.ppc,
                            cells_per_block = self.cpb,
                            transform_sqrt = self.do_sqrt,
                            visualize = True,
                            feature_vector = False)
    return features, hog_img

  def new_frame(self, frame):

    self.RGB_img = frame
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape

    self.hogA, self.hogA_img = self.hogFn(self.ABC_img[:, :, 0])
    self.hogB, self.hogB_img = self.hogFn(self.ABC_img[:, :, 1])
    self.hogC, self.hogC_img = self.hogFn(self.ABC_img[:, :, 2])

  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):

    x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)

    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    hog = np.hstack((hogA, hogB, hogC))

    return hog

  def features(self, frame):
    self.new_frame(frame)
    return self.slice(0, 0, frame.shape[1] , frame.shape[0])######################added *ppc_N

  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):

    if h_pix is None and w_pix is None:
      h_pix, w_pix = self.s, self.s

    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1

    return x_start, x_end, y_start, y_end


sourcer_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             #
  'number_of_orientations': 11,        # 6 - 12
  'pixels_per_cell': 16,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}

start_frame = imread("Data/vehicles/KITTI_extracted/5364.png")

ppc_N = sourcer_params['pixels_per_cell']

sourcer = FeatureSourcer(sourcer_params, start_frame)
print("Loading images to memory...")
t_start = time.time()

vehicle_imgs, nonvehicle_imgs = [], []
vehicle_paths = glob.glob('Data/vehicles/*/*.png')
nonvehicle_paths = glob.glob('Data/non-vehicles/*/*.png')

for path in vehicle_paths: vehicle_imgs.append(imread(path))
for path in nonvehicle_paths: nonvehicle_imgs.append(imread(path))

vehicle_imgs, nonvehicle_imgs = np.asarray(vehicle_imgs), np.asarray(nonvehicle_imgs)
total_vehicles, total_nonvehicles = vehicle_imgs.shape[0], nonvehicle_imgs.shape[0]

print("... Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Vehicle images shape: ", vehicle_imgs.shape)
print("Non-vehicle images shape: ", nonvehicle_imgs.shape)

print("Extracting features... This might take a while...")
t_start = time.time()

vehicles_features, nonvehicles_features = [], []

print("Vehicles...")
for img in vehicle_imgs:
  vehicles_features.append(sourcer.features(img))
  print('█', end = '')

print()
print("Non-Vehicles...")
for img in nonvehicle_imgs:
  nonvehicles_features.append(sourcer.features(img))
  print('█', end = '')

vehicles_features = np.asarray(vehicles_features)
nonvehicles_features = np.asarray(nonvehicles_features)

print()
print("...Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Vehicles features shape: ", vehicles_features.shape)
print("Non-vehicles features shape: ", nonvehicles_features.shape)

def boxBoundaries(box):
    xStart = box[0]
    yStart = box[1]
    xEnd = box[0] + box[2]
    yEnd = box[1] + box[2]
    return xStart , yStart, xEnd, yEnd

def drawBoxes(frame, boxes, color = (255,0,0),thick= 10):

    #take a copy from the original image to draw on it all the boxes we want to draw
    outImage = frame.copy()

    #take every box we want to draw on the image,
    #and draw each one of them indvadully
    for box in boxes:

        #get the start position, and the end postion for each traingle
        xStart , yStart, xEnd, yEnd = boxBoundaries(box)

        #draw the rectangle on the image
        cv2.rectangle(outImage, (xStart , yStart),(xEnd, yEnd), color, thick)

    return outImage

#show images function
def showImages(images,nrows,ncols, width, height, depth = 80):
    fig, ax = plt.subplots(nrows= nrows, ncols=ncols, figsize= (width,height),dpi = depth)
    ax = ax.ravel()

    for i in range(len(images)):
        img= images[i]
        ax[i].imshow(img)

    for i in range(nrows*ncols):
        ax[i].axis('off')
def convert(frame, src_model = "rgb", dest_model = "hls"):

    if src_model == "rgb" and dest_model == "hsv":
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    elif src_model == "rgb" and dest_model == "hls":
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    elif src_model == "rgb" and dest_model == "yuv":
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    elif src_model == "rgb" and dest_model == "ycrcb":
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)
    elif src_model == "hsv" and dest_model == "rgb":
      frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    elif src_model == "hls" and dest_model == "rgb":
      frame = cv2.cvtColor(frame, cv2.COLOR_HLS2RGB)
    elif src_model == "yuv" and dest_model == "yuv":
      frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
    elif src_model == "ycrcb" and dest_model == "ycrcb":
      frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2RGB)
    else:
      raise Exception('ERROR:', 'src_model or dest_model not implemented')

    return frame

def draw_debug_board(img0, hot_windows, heatmap, threshold):

    img1 = np.copy(img0)

    img = np.copy(img0)

    thresh_map = HeatmapThresh(heatmap, threshold=threshold)
    img ,posMinMax , labels =HeatmapCord_Draw(img1,thresh_map)
    # plt.imshow(frameOut)
    bboxes = posMinMax
    # hot_windows = boungingBoxesTotal


    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8);
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8);
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)

    # draw hot_windows in the frame
    img_hot_windows = np.copy(img)
    img_hot_windows = drawBoxes(img_hot_windows, hot_windows, thick=2)

    ymax = 0

    board_x = 5
    board_y = 5
    board_ratio = (img.shape[0] - 3*board_x)//3 / img.shape[0] #0.25
    board_h = int(img.shape[0] * board_ratio)
    board_w = int(img.shape[1] * board_ratio)

    ymin = board_y
    ymax = board_h + board_y
    xmin = board_x
    xmax = board_x + board_w

    offset_x = board_x + board_w

    # draw hot_windows in the frame
    img_hot_windows = cv2.resize(img_hot_windows, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_hot_windows

    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_heatmap

    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_labels

    return img