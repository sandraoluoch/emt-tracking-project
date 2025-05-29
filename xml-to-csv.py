# load packages
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from bioio import BioImage
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from itertools import product
from sklearn.neighbors import KDTree

# xml filepath
xml_filepath = "/allen/aics/users/sandra.oluoch/emt-tracking/emt-data-013025/raw/3500006851/B3_P0/downsampled_2x_fullmovie_bdv-curated-mamut.xml"

# movie filepath
movie_filepath = "/allen/aics/assay-dev/users/Sandi/emt-tracking/emt-data-013025/seg/3500006851/B3_P0/movie/seg_downsampled_2x_fullmovie16bit.tiff"

# read xml file with Beautiful Soup
with open(xml_filepath) as f:
    soup = BeautifulSoup(f, "xml")

# Set up dataframe with important columns
cols = ["Cell ID", "Spot ID","Timepoint", "centroid_z", "centroid_y", "centroid_x"] 
rows = [] 

# find AllSpots and extract cell ID, timepoint, z,y, and x centroids, cell name and radius and add to dataframe 
allspots = soup.find_all('Spot')

for info in allspots:
    frame = info.get("FRAME")
    z_centroid = info.get("POSITION_Z")
    y_centroid = info.get("POSITION_Y")
    x_centroid = info.get("POSITION_X")
    name = info.get("name")
    is_parent = 'Parent' in name
    is_daught = 'Daughter' in name
    spot_id = info.get("ID")
    
    rows.append({"Cell ID": name, 
                 "Spot ID": spot_id,
                 "Timepoint": frame, 
                 "centroid_z": z_centroid, 
                 "centroid_y": y_centroid, 
                 "centroid_x": x_centroid,
                 "Parent": is_parent,
                 "Daughter": is_daught,
                }) 
  
# create dataframe
xml_df = pd.DataFrame(rows, columns=cols)
xml_df.set_index('Cell ID', inplace=False)

# extract track ID and add to dataframe 
allTracks = soup.find_all('Track')

for track in allTracks:
    track_id = eval(track['TRACK_ID'])
    cells = set()
    for edge in track.find_all('Edge'):
        cells.add(eval(edge['SPOT_SOURCE_ID'])) # need spot source ID
        cells.add(eval(edge['SPOT_TARGET_ID'])) # need spot target ID
    for cell in cells:
        xml_df.loc[cell,'Track_ID'] = track_id

# Add parent column and add to dataframe

xml_df['Parent_ID'] = -1

xml_df['Parent?'] = ['yes' if x == 'Mitotic Event' else 'no' for x in xml_df['Cell ID']]

xml_df.loc[xml_df['Parent?'] == 'yes', 'Parent_ID'] = xml_df['Track_ID'] 

print("dataframe created with shape of", xml_df.shape)
print(xml_df.head())


# temporarily empty volume column
xml_df['volume'] = None 

IMG = BioImage(movie_filepath) # load movie from filepath once
print("IMG loaded")
print("Beginning to calculate volumes...")

for t, df_t in xml_df.groupby('Timepoint'): # t is the timepoint and df_t is a mini dataframe with only one timepoint 
    t_int = int(t) # convert 't' variable to integer. Currently is a string
    IMG_at_t = IMG.get_image_data('ZYX', T=t_int, C=0) # load image one timepoint at a time
    
    # Extract regions and create a mapping of labels to area
    regions = regionprops(IMG_at_t) # measures properties of labeled image regions
    cells = {prop.label: prop.area for prop in regionprops(IMG_at_t)} # dictionary where key=seg label and value=area

    # Create KDTree for centroid lookup
    
    # centroid = np.array([[prop.label, np.array(prop.centroid)] for prop in regions], dtype=object)
    # tree = KDTree(np.vstack([c[1] for c in centroid]), leaf_size=10)

    filtered_regions = [prop for prop in regions if prop.label != 0]
    cells = {prop.label: prop.area for prop in filtered_regions}
    centroid = np.array([[prop.label, np.array(prop.centroid)] for prop in filtered_regions], dtype=object)
    tree = KDTree(np.vstack([c[1] for c in centroid]), leaf_size=10)

    for i, row in xml_df.iterrows():  # iterate over xml_df_onetimepoint directly
        obj = [
            int(float(row['centroid_z'])),
            int(float(row['centroid_y'])),
            int(float(row['centroid_x']))
        ]

        lbl = IMG_at_t[obj[0], obj[1], obj[2]]
        volume = cells.get(lbl, 0)

        if volume == 0 or lbl == 0:
            distances, indices = tree.query([obj], k=1)
            nearest_index = indices[0][0]  # indices is 2D array [[0]]
            nearest_label = centroid[nearest_index][0]
            xml_df.at[i, 'label'] = nearest_label
            xml_df.at[i, 'volume'] = cells.get(nearest_label, 0)
        else:
            xml_df.at[i, 'label'] = lbl
            xml_df.at[i, 'volume'] = volume
            
print("All volumes calculated and added to dataframe!")


# save dataframe with area info as a csv file
xml_df.to_csv('./curated_tracks_v4.csv')
print("Done! XML successfully converted to csv")