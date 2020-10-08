#Addresses for reading address information, building footprint coordinates, and parcel outline coordinates
root_path           = '/home/nathan/Insight/Tree Data/'
root_image_path     = root_path + 'SAS Output/'
generate_image_path = '/home/nathan/Insight/Code/generateImages.ipynb'
address_info_path   = root_path + 'LA_County_Parcels_small.csv'
footprint_info_path = root_path + 'Building_Footprints-shp/footprint_nodes.csv'

import pandas as pd
address_info   = pd.read_csv(address_info_path)
footprint_info = pd.read_csv(footprint_info_path)

#perform inner join of all tables by AIN
#master_table = pd.concat([address_info,footprint_info], axis = 1, join = 'inner')
master_table = address_info.merge(footprint_info, how = 'inner', on = ['AIN'])
master_table.head()
print(master_table.columns)


def image_by_address(master_table, address, imsize=1024, classified=False):
    """
    Returns imsize x imsize RGB image centering around lot center location based on address
    Calls generateImages class
    master_table: joined table containing addresses, parcels, and building outlines
    address: string with full address
    imsize: side length of image in pixels
    """
    from generateimages import generateImages as gi

    row = master_table.loc[master_table['SitusFullAddress'] == address]
    center_lat = float(list(row['LAT_LON'])[0].split(',')[0])
    center_lon = float(list(row['LAT_LON'])[0].split(',')[1])

    imfo = gi(root_image_path, center_lat, center_lon)
    image = imfo.generate_image(classified=classified)

    return image, imfo, center_lat, center_lon, row


def map_by_address(image, thresh, class_label=5, imsize=1024):
    """
    Returns imsize x imsize binary image centering around lot center location based on address
    Calls generateImages class; shows estimated tree locations
    image: input image with kmeans classifications
    class_label: class label in kmeans classified image (defaults to trees)
    imsize: side length of image in pixels
    """
    label_image = remove_small_clusters(image, thresh, class_label, class_pick=True)

    return label_image


def remove_small_clusters(im, thresh, tree_class=5, class_pick=False):
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

    return output_map

def structures_by_address(master_table, address):
    """
    Returns list of containing the GPS locations of vertices outlining the main structure at a given address
    """
    row = master_table.loc[master_table['SitusFullAddress'] == address]
    wkt = list(row['WKT'])
    return wkt[0].split('MULTIPOLYGON (((')[1].split(')')[0].split(',')

def parcel_boundaries_by_address():
    pass


def convert_vertices_to_pixels(vertices, image_info, imsize=1024):
    """
    Returns a list of tuples containing (x,y) location in pixels defining a polygon
    vertices: list of strings of lat, lon returned by structures_by_address
    """
    import gdal

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


def structure_intersection_map(tree_map, structure_vertices, imsize=1024):
    """
    Return boolean image showing overlapping regions
    """
    import PIL.ImageDraw as ImageDraw
    import PIL.Image as Image
    import numpy as np

    poly_image = Image.new("RGB", (imsize, imsize))
    draw = ImageDraw.Draw(poly_image)
    draw.polygon(tuple(vertex_coordinates), fill=2)
    overlap = np.array(poly_image)[:, :, 0] * tree_image == 2

    return overlap

#Generate working example
import matplotlib.pyplot as plt

address                                    = '7332 ALPINE WAY LOS ANGELES CA 91042'
label_image,imfo,center_lat,center_lon,row = image_by_address(master_table,address, classified = True)
image,imfo,center_lat,center_lon,row       = image_by_address(master_table,address, classified = False)
tree_image                                 = map_by_address(label_image, 1000)
vertices = structures_by_address(master_table,address)
vertex_coordinates = convert_vertices_to_pixels(vertices, imfo, imsize = 1024)
overlap = structure_intersection_map(tree_image, vertex_coordinates)
overlap = remove_small_clusters(overlap,1000)

plt.imshow(image,cmap="PiYG")
plt.show()

plt.imshow(label_image,cmap="PiYG")
plt.show()

plt.imshow(tree_image,cmap="PiYG")
plt.show()

plt.imshow(overlap,cmap="PiYG")
plt.show()