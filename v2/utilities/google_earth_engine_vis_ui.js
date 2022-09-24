var roi = /* color: #d63000 */ee.Geometry.Polygon(
    [[[-79.14550051251464, 35.98539190185745],
      [-79.14550051251464, 35.856384963779504],
      [-78.9724658445459, 35.856384963779504],
      [-78.9724658445459, 35.98539190185745]]], null, false);

var dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
                  .filterDate('2017-01-01', '2017-12-31')
                  .map(function(image){return image.clip(roi)});
var trueColor432 = dataset.select(['B4', 'B3', 'B2']);

var extractIndex = function(img){
    // apr 2022 development: include commonly used index values that describe the image.
    var nir = img.select('B5');
    var swir = img.select('B6');
    var urbanidx = swir.subtract(nir).divide(swir.add(nir)).rename('UI');
    return img
      .addBands([urbanidx])
      .clip(roi);
  }
var dataset_ui = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
  .filterDate('2017-01-01', '2017-12-31')
  .map(extractIndex);
var ui = dataset_ui.select(['UI']);

var trueColor432Vis = {
  min: 0.0,
  max: 0.4,
};
Map.setCenter(-79.06023403141428,35.9083244977152);
Map.addLayer(trueColor432, trueColor432Vis, 'True Color (432)');
Map.addLayer(ui, trueColor432Vis, 'Urban Index');
Export.image(trueColor432.first(), 'RGB Image');
Export.image(ui.first(), 'Urban Index')
