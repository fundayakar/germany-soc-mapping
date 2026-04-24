/***** 0) Bölge: Germany (GAUL level0) *****/
var region = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Germany"));

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'blue'}, 'Germany');

/***** 1) LUCAS noktaları (CSV asset -> Table) *****/
var lucas_table = ee.FeatureCollection("projects/seismic-relic-481709-r8/assets/LUCAS");

// Germany içinde + OC null değil
var lucas_de = lucas_table
  .filterBounds(region)
  .filter(ee.Filter.neq('OC', null));

print('LUCAS points in Germany (non-null OC):', lucas_de.size());
Map.addLayer(lucas_de, {color: 'red'}, 'LUCAS points (DE)');

/***** 2) Sentinel-2 kompozit (Apr-Oct 2022) *****/
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(region)
  .filterDate('2022-04-01', '2022-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median()
  .clip(region);

// NDVI
var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');

// S2 predictor set
var s2_predictors = s2.addBands(ndvi);

/***** 3) BARE SOIL mask (NDVI < thresh) *****/
var NDVI_THRESH = 0.45;   // 0.35 veya 0.45 deniyorsun, burada değiştir
var bareMask = ndvi.lt(NDVI_THRESH);

// sadece bare-soil pikselleri kalsın
var s2_masked = s2_predictors.updateMask(bareMask);

/***** 4) Sentinel-1 (SAR) kompozit (Apr-Oct 2022) *****/
// IW, GRD, VV+VH, orbit mix, median
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(region)
  .filterDate('2022-04-01', '2022-10-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV','VH'])
  .median()
  .clip(region);

// S1’den basit türevler
var vv = s1.select('VV');
var vh = s1.select('VH');

// log-ratio gibi faydalı olabilen değişkenler
var vv_vh_ratio = vv.divide(vh).rename('VV_div_VH');
var vv_minus_vh  = vv.subtract(vh).rename('VV_minus_VH');

// S1 predictor set
var s1_predictors = s1.addBands([vv_vh_ratio, vv_minus_vh]);

// S1’i de bareMask ile mask’le (aynı pikseller)
var s1_masked = s1_predictors.updateMask(bareMask);

/***** 5) S2 + S1 birleşik predictor set *****/
var predictors_all = s2_masked.addBands(s1_masked);

/***** 6) Noktalardan örnekle -> training tablosu *****/
var training = predictors_all.sampleRegions({
  collection: lucas_de,
  properties: ['OC'],
  scale: 10,
  geometries: false
});

// notNull filtresi (tüm kolonlar dolu olsun)
training = training.filter(ee.Filter.notNull([
  'B2','B3','B4','B8','B11','B12','NDVI',
  'VV','VH','VV_div_VH','VV_minus_VH',
  'OC'
]));

print('Training rows (after notNull, bare soil):', training.size());
print('Training example row:', training.first());

/***** 7) Görsel kontrol *****/
Map.addLayer(ndvi, {min: 0, max: 0.8}, 'NDVI (Apr-Oct 2022)');
Map.addLayer(bareMask.selfMask(), {}, 'Bare mask (NDVI<th)');
Map.addLayer(vv, {min: -20, max: 0}, 'S1 VV (median)');
Map.addLayer(vh, {min: -30, max: -5}, 'S1 VH (median)');

/***** 8) Export CSV *****/
Export.table.toDrive({
  collection: training,
  description: 'DE_LUCAS2018_S2S1_2022_BARESOIL_training',
  fileFormat: 'CSV'
});
