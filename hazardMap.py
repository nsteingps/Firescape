class hazardMap:
    import itertools

    def __init__(self, zip_code):
        import pandas as pd
        import os

        # Addresses for reading address information, building footprint coordinates, and parcel outline coordinates
        self.root_path = os.environ['PWD']
        self.root_image_path = self.root_path + '/data/'
        self.generate_image_path = self.root_path + '/generateImages.ipynb'
        self.master_file_path    = self.root_image_path
        self.master_table = pd.read_csv(self.master_file_path + zip_code + '.csv')

    def image_by_address(self, address, imsize=1024, classified=False):
        """
        Returns imsize x imsize RGB image centering around lot center location based on address
        Calls generateImages class
        master_table: joined table containing addresses, parcels, and building outlines
        address: string with full address
        imsize: side length of image in pixels
        """
        from generateimages import generateImages as gi

        row = self.master_table.loc[self.master_table['SitusFullAddress'] == address]
        center_lat = float(list(row['LAT_LON'])[0].split(',')[0])
        center_lon = float(list(row['LAT_LON'])[0].split(',')[1])

        imfo = gi(self.root_image_path, center_lat, center_lon)
        image = imfo.generate_image(classified=classified)

        return image, imfo, center_lat, center_lon, row

    def map_by_address(self, image, thresh, class_label=5, imsize=1024):
        """
        Returns imsize x imsize binary image centering around lot center location based on address
        Calls generateImages class; shows estimated tree locations
        image: input image with kmeans classifications
        class_label: class label in kmeans classified image (defaults to trees)
        imsize: side length of image in pixels
        """
        label_image, labeled, ncomponents = self.remove_small_clusters(image, thresh, class_label, class_pick=True)

        return label_image

    def remove_small_clusters(self, im, thresh, tree_class=5, class_pick=False):
        """
        Isolates tree-heavy class for kmeans classified image and
        performs connected component analysis to remove components where
        N_pixels < thresh
        returns binary image of large tree clusters
        """
        import numpy as np
        from scipy.ndimage.measurements import label

        if class_pick:
            isolated_class_image = im == tree_class
        else:
            isolated_class_image = im

        isolated_class_image.astype(int)
        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(isolated_class_image, structure)
        unique, counts = np.unique(labeled, return_counts=True)
        chunks = dict(zip(unique, counts))
        chunks = {key: val for key, val in chunks.items() if ((val > thresh) and (val != max(chunks.values())))}
        output_map = np.zeros_like(isolated_class_image)
        keys = [key for key in chunks]

        for i in keys:
            output_map[labeled == i] = 1

        return output_map, labeled, ncomponents

    def structures_by_address(self, address):
        """
        Returns list of containing the GPS locations of vertices outlining the main structure at a given address
        """
        row = self.master_table.loc[self.master_table['SitusFullAddress'] == address]
        wkt = list(row['WKT'])
        return wkt[0].split('MULTIPOLYGON (((')[1].split(')')[0].split(',')

    def parcel_boundaries_by_address(self, address):
        """
        Returns list of containing the GPS locations of vertices outlining the property boundaries at a given address
        """
        row = self.master_table.loc[self.master_table['SitusFullAddress'] == address]
        wkt = list(row['PARCEL_WKT'])
        return wkt[0].split('MULTIPOLYGON (((')[1].split(')')[0].split(',')

    def convert_vertices_to_pixels(self, vertices, image_info, imsize=1024):
        """
        Returns a list of tuples containing (x,y) location in pixels defining a polygon
        vertices: list of strings of lat, lon returned by structures_by_address
        """
        from osgeo import gdal

        center_coordinates = image_info.center_coords
        ul_subset_x = center_coordinates[0] - imsize / 2.  # x pixel location of upper left corner in subset image
        ul_subset_y = center_coordinates[1] - imsize / 2.  # y pixel location of upper left corner in subset image

        vertex_coordinates = []
        driver = gdal.GetDriverByName('GTiff')
        filename = image_info.parent_image_path  # path to raster
        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        transform = dataset.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = -transform[5]
        data = band.ReadAsArray(0, 0, cols, rows)
        points_list = [(float(vertex.split()[0]), float(vertex.split()[1])) for vertex in
                       vertices]  # list of X,Y coordinates

        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1]) / pixelHeight)
            vertex_coordinates.append((col - ul_subset_x, row - ul_subset_y))

        return vertex_coordinates

    def structure_intersection_map(self, tree_map, structure_vertices, imsize=1024):
        """
        Return boolean image showing regions where structures and trees overlap
        """
        import PIL.ImageDraw as ImageDraw
        import PIL.Image as Image
        import numpy as np

        poly_image = Image.new("RGB", (imsize, imsize))
        draw = ImageDraw.Draw(poly_image)
        draw.polygon(tuple(structure_vertices), fill=2)
        overlap = np.array(poly_image)[:, :, 0] * tree_map == 2

        return overlap, True in overlap

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Returns a list of (x,y) tuples of pixel locations connecting the input points
        """
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        switched = False
        if x0 > x1:
            switched = True
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        if y0 < y1:
            ystep = 1
        else:
            ystep = -1

        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = -deltax / 2
        y = y0

        line = []
        for x in range(x0, x1 + 1):
            if steep:
                line.append((y, x))
            else:
                line.append((x, y))

            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        if switched:
            line.reverse()
        return line

    def get_parcel_pixels(self, parcel_coordinates):
        """
        Takes in parcel coordinates returned by convert_vertices_to_pixels
        Returns list of all (x,y) points connecting those vertices in a straight line
        """
        parcel_pixels = [self.bresenham_line(int(parcel_coordinates[i][0]), int(parcel_coordinates[i][1]),
                                        int(parcel_coordinates[i + 1][0]), int(parcel_coordinates[i + 1][1])) for i in
                         range(len(parcel_coordinates) - 1)]
        return list(self.itertools.chain.from_iterable(parcel_pixels))

    def get_parcel_image(self, parcel_pixels, imsize=1024):
        """
        Input: parcel pixels from get_parcel_pixels, image size in pixels
        Returns imsize^2 nummpy array with parcel pixels as 2 and others as 0
        """
        import numpy as np

        parcel_image = np.zeros((imsize, imsize, 3))
        for i in range(len(parcel_pixels)):
            parcel_image[parcel_pixels[i][1], parcel_pixels[i][0], :] = 2

        return parcel_image

    def parcel_intersection_map(self, tree_map, parcel_image, warn_padding, imfo):
        """
        Return boolean image showing regions where property lines and trees overlap
        tree_map: input map of tree locations
        parcel_image: input image with parcel lines
        warn_padding: distance to expand intersection flag (pixels)
        imfo: image info
        """
        import scipy
        import numpy as np

        parcel_image = scipy.ndimage.filters.convolve(parcel_image, np.ones((warn_padding, warn_padding, 3)))
        parcel_image[parcel_image > 0] = 2
        overlap = parcel_image[:, :, 0] * tree_map == 2

        return overlap, True in overlap

    def address_suggestions(self, user_input, n_suggestions=5, thresh=0.2):
        """
        takes in user input string and returns top n_suggestions suggestions
        master_table: master dataframe that includes full addresses as SitusFullAddress
        user_input: user input string
        n_suggestions: maximum number of ranked suggestions returned
        thresh: similarity threshold below which no suggestions are returned
        """
        import difflib

        suggestions = difflib.get_close_matches(user_input.upper(), self.master_table.SitusFullAddress.astype(str),
                                                n=n_suggestions, cutoff=thresh)
        return suggestions

    def generate_flagged_image(self, image, structure_overlap, parcel_overlap, structure_flag, parcel_flag,
                               adjacency_flag=False):
        """
        Generate a color-coded image flagging all potential violations
        image: original satellite image
        structure_overlap: boolean matrix identifying tree-structure overlap
        parcel_overlap: boolean matrix identifying tree-parcel overlap
        structure_flag: boolean denoting whether there is structure overlap in the image
        parcel_flag: boolean denoting whether there is parcel overlap in the image
        """
        import numpy as np

        image = np.array(image)
        if structure_flag:
            new_redband = structure_overlap * 100 + image[:, :, 0]
            image[:, :, 0] = new_redband
        if parcel_flag:
            new_blueband = parcel_overlap * 100 + image[:, :, 2]
            image[:, :, 2] = new_blueband

        return image

    def get_tree_bounds(self, image, conf_thresh=0.3, overlap_thresh=10, imsize=1024, display=False):
        """
        Implements YOLOv3 model trained to detect Southern California trees in Google Earth images
        returns class IDs (currently single-class), confidences, and locations as list of lists [top left x, top left y, w, h]
        image: input RGB image
        conf_thresh: only classifications above this threshold confidence are counted
        overlap_thresh: remove duplicate boxes if their centers are within this euclidean distance (removes dupdetections)
        display: option to display labeled bounding boxes
        """
        import cv2
        import numpy as np
        from scipy.spatial.distance import cdist

        # Load Yolo
        net = cv2.dnn.readNet(self.root_image_path + "yolov3_training_final.weights",
                              self.root_image_path + "yolov3_testing.cfg")

        # Name custom object
        classes = ["tree"]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Load image
        img = np.array(image)
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape

        # Detect trees
        blob = cv2.dnn.blobFromImage(img, 0.00392, (imsize, imsize), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        locs = np.array([[-100, -100]])
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_thresh:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    if min(min(cdist(np.array([[center_x, center_y]]), locs))) > overlap_thresh:
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        locs = np.vstack([locs, [center_x, center_y]])
                        class_ids.append(class_id)

        if display:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

            cv2.imshow("Image", img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

        return class_ids, confidences, boxes

        return class_ids, confidences, boxes

    def trees_on_property(self, boxes, parcel_coordinates):
        """
        returns list of all tree boxes that overlap with the property
        boxes: all tree bounds returned by get_tree_bounds
        parcel_coordinates: pixel vertex coordinates of property boundaries
        """
        import numpy as np
        import matplotlib.path as mpltPath
        valid_boxes = []
        for box in boxes:
            points = [(box[0], box[1]), (box[0], box[1] + box[3]), (box[0] + box[2], box[1]),
                      (box[0] + box[2], box[1] + box[3])]
            path = mpltPath.Path(parcel_coordinates)
            inside2 = np.any(path.contains_points(points))
            if inside2:
                valid_boxes.append(box)

        return valid_boxes

    def get_overlapping_boxes(self, boxes):
        """
        takes in list of box locations and returns list of boxes that overlap another box
        returns list over overlapping pairs of boxes and list of unique boxes that overlap
        """
        import itertools

        # find all pairs of boxes that overlap
        box_combos = list(itertools.combinations(boxes, 2))
        overlapping_boxes = [pair for pair in box_combos if self.intersects(pair)]

        # hone to unique boxes that overlap another box
        unpacked_overlapping_boxes = []
        for pair in overlapping_boxes:
            unpacked_overlapping_boxes.append(pair[0])
            unpacked_overlapping_boxes.append(pair[1])
        unique_boxes_with_overlap = [list(x) for x in set(tuple(x) for x in unpacked_overlapping_boxes)]

        return overlapping_boxes, unique_boxes_with_overlap

    def intersects(self, pair):
        """
        return pairs of boxes that intersect
        """
        b1_top_right_x = pair[0][0] + pair[0][2]
        b1_top_right_y = pair[0][1]
        b1_bot_left_x = pair[0][0]
        b1_bot_left_y = pair[0][1] + pair[0][3]
        b2_top_right_x = pair[1][0] + pair[0][2]
        b2_top_right_y = pair[1][1]
        b2_bot_left_x = pair[1][0]
        b2_bot_left_y = pair[1][1] + pair[0][3]
        return not (b1_top_right_x < b2_bot_left_x or b1_bot_left_x > b2_top_right_x or b1_top_right_y > b2_bot_left_y or b1_bot_left_y < b2_top_right_y)

    def filtered_tree_image(self, boxes, tree_image, thresh=500):
        """
        removes tree pixels if they do not intersect strongly enough with tree bounding boxes
        input: bounding boxes, pixel clasification image labeled by tree only, threshold for minimum number of pixels that must overlap with a box for the cluster to be permitted
        """
        import numpy as np

        tree_boxes = np.zeros_like(tree_image)

        for box in boxes:
            lx = box[0]
            ly = box[1]
            ux = box[0] + box[2]
            uy = box[1] + box[3]
            tree_boxes[ly:uy, lx:ux] = 1

        all_trees, all_trees_labeled, all_trees_ncomponents = self.remove_small_clusters(tree_image, 0)
        all_trees_labeled = all_trees_labeled
        cluster_tree_intersection = tree_boxes * all_trees_labeled

        unique, counts = np.unique(cluster_tree_intersection, return_counts=True)
        chunks = dict(zip(unique, counts))
        chunks = {key: val for key, val in chunks.items() if ((val > thresh) and (val != max(chunks.values())))}
        output_map = np.zeros_like(tree_image)
        keys = [key for key in chunks]
        for i in keys:
            output_map[all_trees_labeled == i] = 1

        return output_map