import streamlit as st
from hazardMap import hazardMap as hm
import PIL.Image as Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os

#user_input_zip     = '91042'
st.image(Image.open(os.environ['PWD'] + '/data/logo.png'), clear_figure = False)

user_input_zip     = st.text_input("zip code")
#user_input_address = '7348 ALPINE WAY LOS ANGELES CA 91042'

if user_input_zip:
    map_instance = hm(user_input_zip)
    user_input_address = st.text_input("address")

    if user_input_address:

        address            = map_instance.address_suggestions(user_input_address)[0]
        label_image,imfo,center_lat,center_lon,row     = map_instance.image_by_address(address, classified = True)
        rgb_image,imfo,center_lat,center_lon,row       = map_instance.image_by_address(address, classified = False)
        class_ids, confidences, boxes                  = map_instance.get_tree_bounds(rgb_image)
        structure_vertices                             = map_instance.structures_by_address(address)
        parcel_vertices                                = map_instance.parcel_boundaries_by_address(address)
        structure_coordinates                          = map_instance.convert_vertices_to_pixels(structure_vertices, imfo, imsize = 1024)
        parcel_coordinates                             = map_instance.convert_vertices_to_pixels(parcel_vertices, imfo, imsize = 1024)
        parcel_pixels                                  = map_instance.get_parcel_pixels(parcel_coordinates)
        parcel_image                                   = map_instance.get_parcel_image(parcel_pixels)
        boxes                                          = map_instance.trees_on_property(boxes, parcel_coordinates)
        tree_image                                     = map_instance.map_by_address(label_image, 1000)
        tree_image                                     = map_instance.filtered_tree_image(boxes, tree_image)
        structure_overlap, structure_flag              = map_instance.structure_intersection_map(tree_image, structure_coordinates)
        structure_overlap,labeled, ncomponents         = map_instance.remove_small_clusters(structure_overlap, 1500)
        parcel_overlap, parcel_flag                    = map_instance.parcel_intersection_map(tree_image, parcel_image, 5, imfo)
        overlapping_boxes, unique_boxes_with_overlap   = map_instance.get_overlapping_boxes(boxes)
        tree_proximity_flag                            = len(unique_boxes_with_overlap) > 0
        all_trees, all_trees_labeled, all_trees_ncomponents = map_instance.remove_small_clusters(tree_image, 0)

        if True in structure_overlap or parcel_flag or tree_proximity_flag:
            st.image(Image.open(os.environ['PWD'] + '/data/potential_hazards.png'), clear_figure = False)
            if True in structure_overlap:
                st.image(Image.open(os.environ['PWD'] + '/data/tree_overhang_warning.png'), clear_figure = False)
            if parcel_flag:
                st.image(Image.open(os.environ['PWD'] + '/data/property_line_warning.png'), clear_figure = False)
            if tree_proximity_flag:
                st.image(Image.open(os.environ['PWD'] + '/data/tree_proximity_warning.png'), clear_figure = False)
            st.image(Image.open(os.environ['PWD'] + '/data/caveat.png'), clear_figure = False)

        final_image = map_instance.generate_flagged_image(rgb_image, structure_overlap, parcel_overlap, structure_flag, parcel_flag)
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, aspect='equal')

        patches = []
        for box in unique_boxes_with_overlap:
            patches.append(matplotlib.patches.Rectangle((box[0],box[1]),box[2],box[3], fill = None, color = 'orange'))
        ax.add_collection(PatchCollection(patches, match_original = True))
        ax.set_axis_off()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        figure = plt.imshow(final_image,cmap="PiYG")
        plt.savefig('figure.png')
        st.image(Image.open('figure.png'), clear_figure = False)

        st.image(Image.open(os.environ['PWD'] + '/data/full_ordinances.png'), clear_figure = False)



