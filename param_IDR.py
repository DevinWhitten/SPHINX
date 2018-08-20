#########################################################################
params = {
"TMIN" : 4000,
"TMAX" : 6500,
"T_ERR_MAX" : 120.,

"FEH_MAX" : -1.0,
"FEH_MIN" : -4.0,
"FEH_ERR_MAX": 0.20,

"CFE_MAX" : 0.00,

"mag_err_max" : 0.10,
"mag_faint_lim" : 19,
"mag_bright_lim" : 14,

"EBV_MAX": 0.05,

"target_var": "TEFF_ADOP",
"format_var": "TEFF",

"target_path" : "~/Google Drive/JPLUS/Databases/IDR/IDR_6_APER_pSL_noEBV.csv.gzip",
"segue_path"  : "datasets/SEGUE_calibrated_catalog.csv",
"edr_segue_path": "datasets/EDR_SEGUE.csv",
"idr_segue_path": "datasets/IDR_SEGUE.csv",
"idr_segue_sup_path": "datasets/IDR_SEGUE_VMP_sup.csv",
"idr_segue_dr10_path": "datasets/IDR_DR10_SEGUE_SUPPLEMENTED.csv",

"output_directory": "output/test/",
"output_filename":"IDR_6_APER_feh_t1.csv",

"target_bands": ['gSDSS_0',     'rSDSS_0',     'iSDSS_0',     'J0395_0',     'J0410_0',     'J0430_0',     'J0515_0',    'J0660_0',    'J0861_0'],
"target_sigma": ['ERR_MAG_6_gSDSS', 'ERR_MAG_6_rSDSS', 'ERR_MAG_6_iSDSS', 'ERR_MAG_6_J0395', "ERR_MAG_6_J0410", 'ERR_MAG_6_J0430', 'ERR_MAG_6_J0515','ERR_MAG_6_J0660','ERR_MAG_6_J0861'],

"format_bands": ['gSDSS', 'rSDSS', 'iSDSS',  'F395',  'F410',  'F430',  'F515',  'F660',  'F861'],

"segue_bands":  ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"segue_sigma":  ["gMag_Sigma", 'rMag_Sigma', 'iMag_Sigma', 'F395Mag_Sigma', 'F410Mag_Sigma', 'F430Mag_Sigma', 'F515Mag_Sigma', 'F660Mag_Sigma', 'F861Mag_Sigma'],

"idr_segue_bands":    ['gSDSS_0',     'rSDSS_0',     'iSDSS_0',      'J0395_0',     'J0410_0',     'J0430_0',     'J0515_0',    'J0660_0',    'J0861_0'],
"idr_segue_sigma": ['ERR_MAG_6_gSDSS',     'ERR_MAG_6_rSDSS',     'ERR_MAG_6_iSDSS',      'ERR_MAG_6_J0395',     'ERR_MAG_6_J0410',     'ERR_MAG_6_J0430',     'ERR_MAG_6_J0515',    'ERR_MAG_6_J0660',    'ERR_MAG_6_J0861']

}
###########################################################################
