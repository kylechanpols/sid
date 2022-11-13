import geojson
import urllib.request
import pyperclip

osm_ref = "912940"

with urllib.request.urlopen(f"http://polygons.openstreetmap.fr/get_geojson.py?id={osm_ref}&params=0") as url:
    data = geojson.load(url)

#data['geometries'][0]['coordinates'][5]
pyperclip.copy(str(data['geometries'][0]['coordinates'][0]))
