/*******************************************************
M4: FINAL SOC MAPPING – Germany (2022)
Model: Random Forest regression
Inputs: Sentinel-2 (Apr–Oct 2022) + NDVI
        Sentinel-1 (2022) + derived radar vars
        DEM (SRTM elev + slope + aspect)
Target: LUCAS OC -> OC_num (numeric)
Export: SOC GeoTIFF to Google Drive
*******************************************************/

// 0) REGION: Germany
var region = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Germany"));
var regionGeom = region.geometry();

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'blue'}, 'Germany');

// 1) LUCAS calibration points (clean OC -> OC_num)
var lucas_raw = ee.FeatureCollection("projects/seismic-relic-481709-r8/assets/LUCAS")
  .filterBounds(regionGeom);

var lucas = lucas_raw.map(function(f){
  var v = ee.String(f.get('OC')); // force to string
  var isNum = v.match('^\\s*[0-9.]+\\s*$').size().gt(0);
  var oc_num = ee.Number(ee.Algorithms.If(isNum, ee.Number.parse(v), null));
  return f.set('OC_num', oc_num);
}).filter(ee.Filter.notNull(['OC_num']))
  .filter(ee.Filter.gt('OC_num', 0)); // keep only positive OC

print('LUCAS points (valid OC_num):', lucas.size());
print('Example lucas row:', lucas.first());
Map.addLayer(lucas, {color: 'red'}, 'LUCAS points');

// 2) Sentinel-2 SR (Apr–Oct 2022) + NDVI
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(regionGeom)
  .filterDate('2022-04-01','2022-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median()
  .clip(regionGeom);

var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');
var s2_stack = s2.addBands(ndvi);

// 3) Sentinel-1 GRD (2022 full year) + derived vars
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(regionGeom)
  .filterDate('2022-01-01','2022-12-31')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
  .select(['VV','VH'])
  .median()
  .clip(regionGeom);

var vv = s1.select('VV');
var vh = s1.select('VH');

var s1_stack = s1.addBands([
  vv.divide(vh).rename('VV_div_VH'),
  vv.subtract(vh).rename('VV_minus_VH')
]);

// 4) DEM (SRTM) + slope + aspect
var dem = ee.Image("USGS/SRTMGL1_003")
  .select('elevation')
  .rename('elev')
  .clip(regionGeom);

var terrain = ee.Terrain.products(dem);
var dem_stack = ee.Image.cat([
  dem,
  terrain.select('slope').rename('slope'),
  terrain.select('aspect').rename('aspect')
]).clip(regionGeom);

// 5) Predictor stack
var predictors = s2_stack.addBands(s1_stack).addBands(dem_stack);

// IMPORTANT: lock feature list (stable band order)
var featureBands = [
  'B2','B3','B4','B8','B11','B12','NDVI',
  'VV','VH','VV_div_VH','VV_minus_VH',
  'elev','slope','aspect'
];

print('Predictor bands:', predictors.bandNames());

// 6) Sample training data
var training = predictors.select(featureBands).sampleRegions({
  collection: lucas,
  properties: ['OC_num'],
  scale: 10,
  geometries: false,
  tileScale: 4
}).filter(ee.Filter.notNull(featureBands.concat(['OC_num'])));

print('Training rows:', training.size());
print('Training example row:', training.first());

// 7) Train RF regression model
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 500,
  seed: 42
}).setOutputMode('REGRESSION');

var trained = rf.train({
  features: training,
  classProperty: 'OC_num',
  inputProperties: featureBands
});

// 8) Predict SOC over Germany
var soc_map = predictors.select(featureBands)
  .classify(trained)
  .rename('SOC');

// 9) Visualize
Map.addLayer(soc_map, {
  min: 0,
  max: 150
}, 'SOC map');

// 10) Export GeoTIFF (Cloud Optimized + compressed + tiled)
Export.image.toDrive({
  image: soc_map.float(),
  description: 'Germany_SOC_Final_Map',
  folder: 'GEE',
  scale: 20,
  region: regionGeom,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,
    fileDimensions: [4096, 4096],
    shardSize: 256
  }
});
