import ee
import geemap
import os
from PIL import ImageDraw, ImageFont
from IPython.display import Image, display
import urllib.request

# GOOGLE EARTH ENGINE - USER INFO
PROJECT_NAME = 'YOUR-PROJECT-NAME'

# ****** INPUT PARAMETERS *********
# Define the starting date for the timelapse (starting from 2018-10-01)
# See details on 'https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED'
START_DATE = '2019-10-01'
# Define the step size expressed in number of months
STEP = 3
# Define the duration of the timelapse expressed in number of months
DURATION = 66
# Define the max pixel probability of clound accepted for the dataset
MAX_CLOUD_PROBABILITY = 30
# Output filename
FILENAME = 'lake_george.gif'
# Define the rectangle of interest for the analysis (only 2 coordinates)
RECTANGLE_OF_INTEREST = [[149.33, -35.22], [149.52, -34.96]]
# [OPTIONAL] Define a specific region where to search the largest water body 
# (list of points defining a closed polyline, i.e. the last point must be equal 
# to the first one).
# To ignore this step, leave this parameter as empty list
# POLYGON_SEARCH_REGION = []
POLYGON_SEARCH_REGION = [
          [149.39077392578125, -34.98095836814947],
          [149.38528076171875, -34.98995944024154],
          [149.37498107910156, -35.02764067937651],
          [149.37086120605468, -35.064180513133216],
          [149.37292114257812, -35.091153281410286],
          [149.377041015625, -35.12261023913553],
          [149.38596740722656, -35.15180939795912],
          [149.39214721679687, -35.17426316190342],
          [149.39764038085937, -35.19446624875818],
          [149.39764038085937, -35.20119949487844],
          [149.42922607421875, -35.20119949487844],
          [149.44158569335937, -35.178753170783644],
          [149.46355834960937, -35.15349364526844],
          [149.4704248046875, -35.130472581320994],
          [149.47729125976562, -35.11362377585869],
          [149.48003784179687, -35.10238930290045],
          [149.47179809570312, -35.09059143969075],
          [149.48278442382812, -35.075420250207],
          [149.49102416992187, -35.05574969458055],
          [149.4711114501953, -35.03719882960281],
          [149.45325866699218, -35.02539154040507],
          [149.42373291015625, -35.0270784004351],
          [149.42235961914062, -35.0102082338579],
          [149.41205993652343, -35.004584071596824],
          [149.41480651855468, -34.99614710323569],
          [149.41137329101562, -34.98095836814947],
          [149.39077392578125, -34.98095836814947]
          ]

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project=PROJECT_NAME)

# Load Sentinel-2 data and relative cloud probability data
s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

# Define the region of interest
region = ee.Geometry.Rectangle(RECTANGLE_OF_INTEREST)

# Define profile of the region to find the largest polygon
if not POLYGON_SEARCH_REGION:
  profile = region
else:
  profile = ee.Geometry.Polygon(coords=POLYGON_SEARCH_REGION)

# Function to mask S2 data with cloud probability data
def maskClouds(img):
    clouds = ee.Image(img.get('cloud_mask')) \
                .select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)

# Function to calculate the SWM index, not used in this case
def calculateSWM(img):
    return img.select('B2').add(img.select('B3')) \
            .divide(img.select('B8').add(img.select('B11')))

# Function to calculate the AWEI_ns index
# 4 * (Green - SWIR1) * (0.25 * NIR + 2.75 * SWIR1)
def calculateAWEI(img):
    gr = img.select('B3')
    nir = img.select('B8')
    swir1 = img.select('B11')
    return gr.subtract(swir1).multiply(4) \
            .subtract(nir.multiply(0.25).add(swir1.multiply(2.75))) \
            .rename('AWEI')

# Function to calculate the area of your feature
def calculate_feature_area(feature):
  area = feature.geometry().area(maxError=10) # Calculate area
  return feature.set('area', area) # Set area as a property

# Function to generate the main collection of data (mapping section)
def generate_img_collection(idx):
  start = ee.Date(START_DATE).advance(idx, 'month')
  end = start.advance(STEP, 'month')

  # Join S2 with cloud probability and filter
  criteria = ee.Filter.And(
      ee.Filter.bounds(region),
      ee.Filter.date(start, end)
  )
  s2_with_mask = ee.Join.saveFirst('cloud_mask').apply(
      primary = s2Collection.filter(criteria), \
      secondary = s2Clouds.filter(criteria), \
      condition = ee.Filter.equals(leftField = 'system:index', \
                                  rightField = 'system:index') \
  )

  # Apply mask to remove areas with probability of clouds, make the median
  # and create property of median time.
  s2_processed = ee.ImageCollection(s2_with_mask).map(maskClouds) \
                        .median().divide(10000) \
                        .set('med_time', start.format('YYYY-MM'))

  # Calculate and add AWEI mask
  s2_awei_mask = calculateAWEI(s2_processed).rename('AWEI_MASK').gte(0.2)
  s2_awei_mask = s2_awei_mask.updateMask(s2_awei_mask)
  s2_processed = s2_processed.addBands(s2_awei_mask)

  # Generate polygons corresponding to the water bodies
  water_polygons = s2_awei_mask.reduceToVectors(
      geometry=profile,
      crs=s2_processed.projection(),
      scale=30,
      geometryType='polygon',
      eightConnected=False,
      labelProperty='water',
      reducer=ee.Reducer.countEvery()
  )

  # Count the number of polygons identified
  featureCount = water_polygons.size()

  # Function to generate an image of the lake's boundaries and add it to the
  # featureCollection as band. It also add a property of the area in m^2.
  def polygon2Band(water_polygons, s2_processed):
    # Calculate the area of each polygon and add it as a property
    water_polygons = water_polygons.map(calculate_feature_area)

    # Sort the features by area in descending order
    # False in "sort" for descending order
    largest_polygon = water_polygons.sort('area', False).first()

    # Create a base image with a default value (e.g., 0)
    # + Rasterize the boundary with a specified color (1) and width (1)
    # + create the mask of values above threshold
    poly_image = ee.Image(0).clip(largest_polygon) \
                    .paint(ee.FeatureCollection([largest_polygon]), 1, 1) \
                    .rename('BOUNDARY') \
                    .gt(0)

    # Rasterize the boundary with a specified color and width
    poly_image = poly_image.updateMask(poly_image)
    s2_processed = s2_processed.addBands(poly_image)
    s2_processed = s2_processed.set('area', largest_polygon.get('area'))
    return s2_processed

  # Function to generate an empty image and set the property area to zero.
  def emptyPolygon2Band(s2_processed):
    # Create an empty Band with a specified color (0) and width (0)
    poly_image = ee.Image(0).clip(region) \
                            .paint(ee.FeatureCollection([region]), 0, 0) \
                            .rename('BOUNDARY') \
                            .gt(0)
    # Rasterize the boundary with a specified color and width
    poly_image = poly_image.updateMask(poly_image)
    s2_processed = s2_processed.addBands(poly_image)
    s2_processed = s2_processed.set('area', 0)
    return s2_processed

  # If there are polygons, generate image of the boundaries of the largest one
  # Otherwise, generate an empty image.
  s2_processed = ee.Algorithms.If(featureCount.gt(0),
                                  polygon2Band(water_polygons, s2_processed),
                                  emptyPolygon2Band(s2_processed))

  return s2_processed

# Generate main collection of data with given filters and required bands
month_idx = ee.List.sequence(0, DURATION, STEP)
collection = ee.ImageCollection.fromImages(month_idx.map(generate_img_collection))

# Get list of 'med_time'
med_time_list = collection.reduceColumns(reducer=ee.Reducer.toList(), \
                                selectors=['med_time']).get('list').getInfo()

# Get list of 'area'
area_list = collection.reduceColumns(reducer=ee.Reducer.toList(), \
                                selectors=['area']).get('list').getInfo()

# Function to visualise the satellite image
def visualize_image_rgb(image):
  vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
  return image.visualize(**vis_params)

# Function to visualise the satellite image and the water boundary
def visualize_image_awei(image):
  vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
  vis_params_poly = {'bands': ['BOUNDARY'], 'palette': ['white', 'red']}
  output = image.visualize(**vis_params)
  output = output.blend(image.visualize(**vis_params_poly))
  return output

# Create RGB visualization images for use as animation frames.
rgbVis = collection.map(visualize_image_awei)

# Define thumbnail parameters
thumb_params = {
    'region': region,
    'dimensions': 520,
    'crs': 'EPSG:3857',
    'framesPerSecond': 1.5,
    'format': 'gif'
}

# Get URL that will produce the animation when accessed.
gif_url = rgbVis.getVideoThumbURL(thumb_params)
#print(gif_url)
#display(Image(url=gif_url))

# Save gif locally
gif_path = os.path.join(os.getcwd(), FILENAME)
urllib.request.urlretrieve(gif_url, gif_path)

# Generate label of the gif frames
label = [f'{x}, {y/1e6:06.2f} km^2' for x,y in zip(med_time_list, area_list)]

# Add labels to the gif frames
geemap.add_text_to_gif(
    gif_path,
    gif_path,
    xy=("5%", "3%"),
    text_sequence=label,
    font_size=24,
    font_color="#ffffff",
    duration=666, # in ms
)