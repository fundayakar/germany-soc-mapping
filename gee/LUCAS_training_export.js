/*******************************************************
Germany | LUCAS OC + Sentinel-2 (Apr-Oct 2022) + NDVI
+ Sentinel-1 (2022 full year) + derived vars
+ DEM (SRTM elevation + slope + aspect)
Export: training table CSV to Drive
*******************************************************/

// 0) REGION: Germany
var region = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Germany"));

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'blue'}, 'Germany');

// 1) LUCAS points (non-null OC)
var lucas = ee.FeatureCollection("projects/seismic-relic-481709-r8/assets/LUCAS")
  .filterBounds(region)
  .filter(ee.Filter.neq('OC', null));

print('LUCAS points (non-null OC):', lucas.size());
Map.addLayer(lucas, {color: 'red'}, 'LUCAS points');

// 2) Sentinel-2 SR (Apr–Oct 2022)
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(region)
  .filterDate('2022-04-01', '2022-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median()
  .clip(region);

var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');
var s2_stack = s2.addBands(ndvi);

print('S2 bands:', s2_stack.bandNames());

// 3) Sentinel-1 GRD (2022 full year)
var s1col = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(region)
  .filterDate('2022-01-01', '2022-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV','VH']);

var s1 = s1col.median().clip(region);
var vv = s1.select('VV');
var vh = s1.select('VH');

var vv_div_vh = vv.divide(vh).rename('VV_div_VH');
var vv_minus_vh = vv.subtract(vh).rename('VV_minus_VH');

var s1_stack = s1.addBands([vv_div_vh, vv_minus_vh]);
print('S1 bands:', s1_stack.bandNames());

// 4) DEM (SRTM) + slope + aspect
var dem = ee.Image("USGS/SRTMGL1_003").select('elevation').clip(region).rename('elev');

var terrain = ee.Terrain.products(dem);
var slope = terrain.select('slope').rename('slope');
var aspect = terrain.select('aspect').rename('aspect');

var dem_stack = ee.Image.cat([dem, slope, aspect]).clip(region);
print('DEM bands:', dem_stack.bandNames());

// 5) Combine predictors (S2 + S1 + DEM)
var predictors = s2_stack.addBands(s1_stack).addBands(dem_stack);
print('ALL predictor bands:', predictors.bandNames());

// 6) Sample points
var training = predictors.sampleRegions({
  collection: lucas,
  properties: ['OC'],
  scale: 10,
  geometries: false,
  tileScale: 4
});

// Require full columns
var required = [
  'B2','B3','B4','B8','B11','B12','NDVI',
  'VV','VH','VV_div_VH','VV_minus_VH',
  'elev','slope','aspect',
  'OC'
];

training = training.filter(ee.Filter.notNull(required));

print('Training rows (final):', training.size());
print('Training example row:', training.first());

// 7) Export
Export.table.toDrive({
  collection: training,
  description: 'DE_LUCAS2018_S1S2_DEM_2022_training',
  fileFormat: 'CSV'
});
