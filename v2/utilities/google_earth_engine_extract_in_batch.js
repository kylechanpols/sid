// Meant to be run on the Google Earth Engine IDE

var batch = require('users/fitoprincipe/geetools:batch')

var roi = /* color: #d63000 */ee.Geometry.Polygon(
[[[2.2573333658616734, 41.45860511869924],
          [2.2243743814866734, 41.4658090578404],
          [2.1721893228929234, 41.44625364688157],
          [2.1364837564866734, 41.43595895724499],
          [2.1145111002366734, 41.41124503936027],
          [2.0788055338304234, 41.38136983994822],
          [2.0568328775804234, 41.346326258316026],
          [2.0623260416429234, 41.326734860941585],
          [2.0485931314866734, 41.29888431786033],
          [2.0815521158616734, 41.27411829126909],
          [2.1639495767991734, 41.30197941021872],
          [2.2024017252366734, 41.37827851836506],
          [2.2641998209397984, 41.442135966999196]]]);


var exportRGB = function(img){
  // get the RGB bands of the image in the roi
  return img
    .clip(roi)
    .visualize({bands:['B4','B3','B2'], min:0 , max:0.3, gamma:1.4});
}

var extractIndex = function(img){
  // get the required feature bands of the image in the roi.
  var red = img.select('B4');
  var green = img.select('B3');
  var blue = img.select('B2');
  var nir = img.select('B5');
  var swir = img.select('B6');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var urbanidx = swir.subtract(nir).divide(swir.add(nir)).rename('UI');
  var evi = nir.subtract(red).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).multiply(2.5).rename('EVI');
  
  return img
    .addBands([ndvi, ndwi, urbanidx, evi])
    .select(['B4','B3','B2','NDVI','NDWI','UI','EVI'])
    .toFloat()
    .clip(roi);
}

// generate a list of images before the mapped function
// contact: https://www.cpc.unc.edu/people/fellows/conghe-song/

var dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
  .filterBounds(roi)
  .filterDate('2014-01-01', '2021-12-30')
  .sort('CLOUD_COVER')
  .map(extractIndex)
  .aside(print);

/*
// These are for the Code Editor - it will show the photo to be exported

var takeOne = dataset.first();

print("rgb",takeOne);


var visParams = {bands:["vis-red","vis-green","vis-blue"], min: 0, max: 255};
Map.addLayer(takeOne);
Map.centerObject(roi, 8);

*/

batch.Download.ImageCollection.toDrive(dataset, 'sidv2', 
              {scale: 10, 
               region: roi,
              type: "float"
              });
              
