/*******************************************************
 M5: S1 + S2 (spring + autumn ayrı) + DEM + TWI
 Germany | LUCAS OC | 2022
 Export: training CSV to Drive
*******************************************************/

// 0) Region
var region = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Germany"));
var geom = region.geometry();
Map.centerObject(region, 6);

// 1) LUCAS points
var lucas = ee.FeatureCollection("projects/seismic-relic-481709-r8/assets/LUCAS")
  .filterBounds(geom)
  .filter(ee.Filter.neq('OC', null));
print('LUCAS points:', lucas.size());
Map.addLayer(lucas, {color: 'red'}, 'LUCAS');

// 2) S2 spring composite (Mar-May 2022)
var s2_spring = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geom)
  .filterDate('2022-03-01', '2022-05-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median().clip(geom);

var ndvi_sp = s2_spring.normalizedDifference(['B8','B4']).rename('NDVI_sp');
var bsi_sp  = s2_spring.expression(
  '((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))',
  {B11:s2_spring.select('B11'),B4:s2_spring.select('B4'),
   B8:s2_spring.select('B8'),B2:s2_spring.select('B2')}
).rename('BSI_sp');

var spring_stack = s2_spring.rename(['B2_sp','B3_sp','B4_sp','B8_sp','B11_sp','B12_sp'])
  .addBands(ndvi_sp).addBands(bsi_sp);

// 3) S2 autumn composite (Sep-Nov 2022)
var s2_autumn = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geom)
  .filterDate('2022-09-01', '2022-11-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median().clip(geom);

var ndvi_au = s2_autumn.normalizedDifference(['B8','B4']).rename('NDVI_au');
var bsi_au  = s2_autumn.expression(
  '((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))',
  {B11:s2_autumn.select('B11'),B4:s2_autumn.select('B4'),
   B8:s2_autumn.select('B8'),B2:s2_autumn.select('B2')}
).rename('BSI_au');

var autumn_stack = s2_autumn.rename(['B2_au','B3_au','B4_au','B8_au','B11_au','B12_au'])
  .addBands(ndvi_au).addBands(bsi_au);

// 4) S1 (2022 full year, DESCENDING)
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(geom)
  .filterDate('2022-01-01','2022-12-31')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
  .select(['VV','VH'])
  .median().clip(geom);

var vv = s1.select('VV');
var vh = s1.select('VH');
var s1_stack = s1
  .addBands(vv.divide(vh).rename('VV_div_VH'))
  .addBands(vv.subtract(vh).rename('VV_minus_VH'));

// 5) DEM + slope + aspect + TWI
var dem = ee.Image("USGS/SRTMGL1_003")
  .select('elevation').rename('elev').clip(geom);

var terrain = ee.Terrain.products(dem);
var slope_rad = terrain.select('slope')
  .multiply(Math.PI/180).rename('slope_rad');
var slope = terrain.select('slope').rename('slope');
var aspect = terrain.select('aspect').rename('aspect');

// TWI = ln(upslope_area / tan(slope))
// GEE'de flow accumulation yok ama iyi bir proxy:
// TWI_proxy = elev * cos(aspect_rad) - kullanmak yerine
// flow accumulation olmadan TWI hesaplanamaz gerçek anlamda
// Bu yüzden slope + aspect + elev'i ayrı tutuyoruz
// ve curvature ekliyoruz (TWI'nın bileşenleri)
var curvature = terrain.select('hillshade').rename('hillshade');

var dem_stack = ee.Image.cat([dem, slope, aspect, curvature]);

// 6) Combine all
var predictors = spring_stack
  .addBands(autumn_stack)
  .addBands(s1_stack)
  .addBands(dem_stack);

var featureBands = [
  'B2_sp','B3_sp','B4_sp','B8_sp','B11_sp','B12_sp','NDVI_sp','BSI_sp',
  'B2_au','B3_au','B4_au','B8_au','B11_au','B12_au','NDVI_au','BSI_au',
  'VV','VH','VV_div_VH','VV_minus_VH',
  'elev','slope','aspect','hillshade'
];

print('Predictor bands:', predictors.bandNames());

// 7) Sample
var training = predictors.select(featureBands).sampleRegions({
  collection: lucas,
  properties: ['OC'],
  scale: 10,
  geometries: false,
  tileScale: 4
}).filter(ee.Filter.notNull(featureBands.concat(['OC'])));

print('Training rows:', training.size());
print('Example row:', training.first());

// 8) Export
Export.table.toDrive({
  collection: training,
  description: 'DE_LUCAS_M5_spring_autumn_DEM',
  fileFormat: 'CSV'
});