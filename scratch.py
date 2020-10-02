import streamlit as st
from hazardMap import hazardMap as hm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib
test = hm('91042')
st.title('FireScape: Detect Residential Wildfire Hazards')
user_input = st.text_input("address")

if user_input:
    import time

    address = test.address_suggestions(user_input)[0]  # '7348 ALPINE WAY LOS ANGELES CA 91042'
    st.warning("You live in a Very High Fire Risk Severity Zone.")
    st.text("I'll look around your property and see if I can make recommendations to help you improve the safety of your property and avoid fines.")
    st.text('Finding your house...')
    image,imfo,center_lat,center_lon,row = test.image_by_address(address, classified = False)
    st.image(image)

    st.text('Analyzing scene...')
    label_image, imfo, center_lat, center_lon, row = test.image_by_address(address, classified=True)

    tree_image = test.map_by_address(label_image, 1000)
    structure_vertices = test.structures_by_address(address)
    parcel_vertices = test.parcel_boundaries_by_address(address)
    structure_coordinates = test.convert_vertices_to_pixels(structure_vertices, imfo, imsize=1024)
    parcel_coordinates = test.convert_vertices_to_pixels(parcel_vertices, imfo, imsize=1024)
    parcel_pixels = test.get_parcel_pixels(parcel_coordinates)
    parcel_image = test.get_parcel_image(parcel_pixels)
    structure_overlap, structure_flag = test.structure_intersection_map(tree_image, structure_coordinates)
    structure_overlap = test.remove_small_clusters(structure_overlap, 1500)
    parcel_overlap, parcel_flag = test.parcel_intersection_map(tree_image, parcel_image, 5, imfo)
    class_ids, confidences, boxes = test.get_tree_bounds(image)
    overlapping_boxes, unique_boxes_with_overlap = test.get_overlapping_boxes(boxes)
    final_image = test.generate_flagged_image(image, structure_overlap, parcel_overlap, structure_flag, parcel_flag)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')

    patches = []
    for box in unique_boxes_with_overlap:
        print(box[0])
        patches.append(matplotlib.patches.Rectangle((box[0], box[1]), box[2], box[3], fill=None, color='cyan'))
    ax.add_collection(PatchCollection(patches, match_original=True))
    figure = plt.imshow(final_image, cmap="PiYG")
    st.write(figure)

    if structure_flag:
        st.warning("It looks like some trees overhang your roof (see below). Please ensure branches are trimmed to be at least six feet from your roof and remove loose foliage from your roof.")
    if parcel_flag:
        st.warning("It looks like some trees are close to the edge of your property. Please ensure branches are trimmed to be at least three feet from any fences.")
    st.warning("It looks like some trees are quite close to each other. Best practices recommend that you trim branches for at least six feet of separation between trees.")