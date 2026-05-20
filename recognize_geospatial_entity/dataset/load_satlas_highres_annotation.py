import os
import json

SATLAS_ROOT = "/media/disk5/dataset/satlas/dataset"
polyline_categories = [
    'airport_runway', 'airport_taxiway', 'raceway', 'road', 'railway', 'river',
]

raster_tasks = [{
    'name': 'land_cover',
    'id': 'land_cover',
    'type': 'segment',
    'categories': ['invalid', 'water', 'developed', 'tree', 'shrub', 'grass', 'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss'],
    'image_type': 'all',
    'colors': [
        [0, 0, 0], # (black) invalid
        [0, 0, 255], # (blue) water
        [255, 0, 0], # (red) developed
        [0, 192, 0], # (dark green) tree
        [200, 170, 120], # (brown) shrub
        [0, 255, 0], # (green) grass
        [255, 255, 0], # (yellow) crop
        [128, 128, 128], # (grey) bare
        [255, 255, 255], # (white) snow
        [0, 255, 255], # (cyan) wetland
        [255, 0, 255], # (pink) mangroves
        [128, 0, 128], # (purple) moss
    ],
}, {
    'name': 'crop_type',
    'id': 'crop_type',
    'type': 'segment',
    'categories': ['invalid', 'rice', 'grape', 'corn', 'sugarcane', 'tea', 'hop', 'wheat', 'soy', 'barley', 'oats', 'rye', 'cassava', 'potato', 'sunflower', 'asparagus', 'coffee'],
    'image_type': 'all',
}, {
    'name': 'water_event',
    'id': 'water_event',
    'type': 'segment',
    'categories': ['invalid', 'background', 'water_event'],
    'image_type': 'highres',
    'colors': [
        [0, 0, 0], # (black) invalid
        [0, 255, 0], # (green) background
        [0, 0, 255], # (blue) water_event
    ],
},  {
    'name': 'polyline_bin_segment',
    'type': 'bin_segment',
    'categories': polyline_categories,
    'image_type': 'all',
    'colors': [
        [255, 255, 255], # (white) airport_runway
        [192, 192, 192], # (light grey) airport_taxiway
        [160, 82, 45], # (sienna) raceway
        [255, 255, 255], # (white) road
        [144, 238, 144], # (light green) railway
        [0, 0, 255], # (blue) river
    ]
}]

def process(job):
    satlas_tile, event_id, event_path, is_static = job

    anchor_image_name = None
    # if not is_static:
    #     vector_fname = os.path.join(event_path, 'vector.json')
    #     with open(vector_fname, 'r') as f:
    #         data = json.load(f)

    #     print(data)
    #     exit(0)

    #     # Some dynamic labels don't have ImageName, but ImageName is required.
    #     if 'ImageName' not in data['metadata']:
    #         return

    #     anchor_image_name = data['metadata']['ImageName']


def main():
    jobs = []
    print('populate static jobs')
    static_path = os.path.join(SATLAS_ROOT, 'static')
    for tile_str in os.listdir(static_path):
        parts = tile_str.split('_')
        satlas_tile = (int(parts[0]), int(parts[1]))
        tile_path = os.path.join(static_path, tile_str)
        jobs.append((satlas_tile, tile_str, tile_path, True))
    
    for job in jobs:
        process(job)

if __name__ == '__main__':
    main()