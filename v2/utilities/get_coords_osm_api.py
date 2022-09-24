import geojson
import urllib.request
import pyperclip

osm_ref = "398021"

with urllib.request.urlopen(f"http://polygons.openstreetmap.fr/get_geojson.py?id={osm_ref}&params=0") as url:
    data = geojson.load(url)

pyperclip.copy(str(data['geometries'][0]['coordinates'][0]))
