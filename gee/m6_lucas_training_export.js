/*******************************************************
 M6: S1 + S2 (Apr-Oct) + DEM + ERA5 seasonal
 Germany | LUCAS OC | 2022
 ERA5: kış yağışı (Dec-Feb), yaz sıcaklığı (Jun-Aug),
       yıllık toprak nemi (sm1)
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

// 2) S2 (Apr-Oct 2022)
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geom)
  .filterDate('2022-04-01', '2022-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11','B12'])
  .median().clip(geom);

var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');
var s2_stack = s2.addBands(ndvi);

// 3) S1 (2022 full year, DESCENDING)
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(geom)
  .filterDate('2022-01-01', '2022-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV','VH'])
  .median().clip(geom);

var vv = s1.select('VV');
var vh = s1.select('VH');
var s1_stack = s1
  .addBands(vv.divide(vh).rename('VV_div_VH'))
  .addBands(vv.subtract(vh).rename('VV_minus_VH'));

// 4) DEM + slope + aspect
var dem = ee.Image("USGS/SRTMGL1_003")
  .select('elevation').rename('elev').clip(geom);
var terrain = ee.Terrain.products(dem);
var dem_stack = ee.Image.cat([
  dem,
  terrain.select('slope').rename('slope'),
  terrain.select('aspect').rename('aspect')
]);

// 5) ERA5 mevsimsel
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR");

// Kış yağışı: Dec 2021 + Jan-Feb 2022
var tp_winter = era5
  .filterDate('2021-12-01', '2022-02-28')
  .select('total_precipitation_sum')
  .sum().rename('tp_winter').clip(geom);

// Yaz sıcaklığı: Jun-Aug 2022 ortalaması (Celsius)
var t2m_summer = era5
  .filterDate('2022-06-01', '2022-08-31')
  .select('temperature_2m')
  .mean()
  .subtract(273.15)
  .rename('t2m_summer').clip(geom);

// Yıllık toprak nemi: 2022 ortalaması
var sm_annual = era5
  .filterDate('2022-01-01', '2022-12-31')
  .select('volumetric_soil_water_layer_1')
  .mean().rename('sm_annual').clip(geom);

var era5_stack = ee.Image.cat([tp_winter, t2m_summer, sm_annual]);
print('ERA5 bands:', era5_stack.bandNames());

// 6) Combine
var predictors = s2_stack
  .addBands(s1_stack)
  .addBands(dem_stack)
  .addBands(era5_stack);

var featureBands = [
  'B2','B3','B4','B8','B11','B12','NDVI',
  'VV','VH','VV_div_VH','VV_minus_VH',
  'elev','slope','aspect',
  'tp_winter','t2m_summer','sm_annual'
];

print('All predictor bands:', predictors.bandNames());

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
  description: 'DE_LUCAS_M6_S1S2_DEM_ERA5seasonal',
  fileFormat: 'CSV'
});
