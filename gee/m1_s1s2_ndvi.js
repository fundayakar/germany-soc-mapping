/*******************************************************
 CLEAN + FIXED PIPELINE
 Germany | LUCAS OC + Sentinel-2 (Apr-Oct 2022) + NDVI
 + Sentinel-1 (wider window for overlap) + derived vars
 Exports: training table to Drive (CSV)
*******************************************************/

// ----------------------------
// 0) REGION: Germany
// ----------------------------
var region = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Germany"));

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'blue'}, 'Germany');

// ----------------------------
// 1) LUCAS points (Asset Table)
// ----------------------------
var lucas_table = ee.FeatureCollection("projects/seismic-relic-481709-r8/assets/LUCAS");

// Keep only points inside Germany and with non-null OC
var lucas_de = lucas_table
  .filterBounds(region)
  .filter(ee.Filter.neq('OC', null));

print('LUCAS points in Germany (non-null OC):', lucas_de.size());
Map.addLayer(lucas_de, {color: 'red'}, 'LUCAS points (DE)');

// ----------------------------
// 2) Sentinel-2 SR (Apr-Oct 2022) -> median composite
// ----------------------------
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(region)
  .filterDate('2022-04-01', '2022-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median()
  .clip(region);

// NDVI
var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');

// S2 predictors
var s2_predictors = s2.addBands(ndvi);

// Quick sanity
print('S2 bands:', s2.bandNames());
print('S2_predictors bands:', s2_predictors.bandNames());
Map.addLayer(ndvi, {min: 0, max: 0.8}, 'NDVI (Apr-Oct 2022)');

// ----------------------------
// 3) Sentinel-1 GRD (WIDER window) -> overlap-friendly
//    Use a wider date range to avoid "0 rows" intersection
// ----------------------------
var s1col = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(region)
  .filterDate('2022-01-01', '2022-12-31') // wider window
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select(['VV','VH']);

var s1 = s1col.median().clip(region);

// Derived S1 features (ratio and difference in linear scale is tricky)
// We keep simple robust transforms: VV, VH, VV/VH, VV-VH (in dB space)
var vv = s1.select('VV');
var vh = s1.select('VH');
var vv_div_vh = vv.divide(vh).rename('VV_div_VH');
var vv_minus_vh = vv.subtract(vh).rename('VV_minus_VH');

var s1_predictors = s1.addBands(vv_div_vh).addBands(vv_minus_vh);

print('S1 bands:', s1_predictors.bandNames());

// ----------------------------
// 4) Combine predictors (S2+NDVI + S1 features)
// ----------------------------
var predictors = s2_predictors.addBands(s1_predictors);
print('All predictor bands:', predictors.bandNames());

// ----------------------------
// 5) DEBUG: sample S2 only, S1 only, then BOTH
// ----------------------------
var t_s2 = s2_predictors.sampleRegions({
  collection: lucas_de,
  properties: ['OC'],
  scale: 10,
  geometries: false
}).filter(ee.Filter.notNull(['B2','B3','B4','B8','B11','B12','NDVI','OC']));

print('After S2+NDVI notNull:', t_s2.size());

var t_s1 = s1_predictors.sampleRegions({
  collection: lucas_de,
  properties: ['OC'],
  scale: 10,
  geometries: false
}).filter(ee.Filter.notNull(['VV','VH','VV_div_VH','VV_minus_VH','OC']));

print('After S1 notNull:', t_s1.size());

// ----------------------------
// 6) FINAL TRAINING TABLE (BOTH)
// ----------------------------
var training = predictors.sampleRegions({
  collection: lucas_de,
  properties: ['OC'],
  scale: 10,
  geometries: false
});

// Keep only fully populated rows
var required = ['B2','B3','B4','B8','B11','B12','NDVI','VV','VH','VV_div_VH','VV_minus_VH','OC'];
training = training.filter(ee.Filter.notNull(required));

print('Training rows (final, after notNull):', training.size());
print('Training example row:', training.first());

// If still 0, it will be obvious here:
print('If training is 0, problem = intersection/masking/date. Try: CLOUDY<40 or S2 date wider.');

// ----------------------------
// 7) Export to Drive (CSV)
// ----------------------------
Export.table.toDrive({
  collection: training,
  description: 'DE_LUCAS2018_S2_2022_S1_2022_training',
  fileFormat: 'CSV'
});
