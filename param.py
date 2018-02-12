###########################################################################
params = {
"TMIN" : 5000,
"TMAX" : 8000,
"T_ERR_MAX" : 120.,

"FEH_MAX" : -0.0,
"FEH_MIN" : -4.0,
"FEH_ERR_MAX": 0.20,

"CFE_MAX" : 0.00,

"mag_err_max" : 0.10,
"mag_faint_lim" : 20,
"mag_bright_lim" : 14,

"EBV_MAX": 0.05,

"target_var": "TEFF_ADOP",
"format_var": "TEFF",

"target_path" : "/Users/MasterD/Google Drive/JPLUS/Databases/EDR/EDR_DR10_SEGUE.csv",
"segue_path"  : "/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/data/catalogs/SEGUE_calibrated_catalog.csv",
"idr_segue_path": "Datasets/IDR_SEGUE_SUPPLEMENTED.csv",
"idr_segue_dr10_path": "Datasets/IDR_DR10_SEGUE_SUPPLEMENTED.csv",
"output_directory": "output/",


"target_bands": ['gSDSS_0',     'rSDSS_0',     'iSDSS_0',     'J0395_0',     'J0410_0',     'J0430_0',     'J0515_0',    'J0660_0',    'J0861_0'],
"target_sigma": ['gSDSS_sig', 'rSDSS_sig', 'iSDSS_sig', 'J0395_sig', "J0410_sig", 'J0430_sig', 'J0515_sig','J0660_sig','J0861_sig'],

"format_bands": ['gSDSS', 'rSDSS', 'iSDSS',  'F395',  'F410',  'F430',  'F515',  'F660',  'F861'],

"segue_bands":  ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"segue_sigma":  ["gMag_Sigma", 'rMag_Sigma', 'iMag_Sigma', 'F395Mag_Sigma', 'F410Mag_Sigma', 'F430Mag_Sigma', 'F515Mag_Sigma', 'F660Mag_Sigma', 'F861Mag_Sigma'],

"idr_segue_bands":    ['gSDSS_0',     'rSDSS_0',     'iSDSS_0',      'J0395_0',     'J0410_0',     'J0430_0',     'J0515_0',    'J0660_0',    'J0861_0'],
"idr_segue_sigma": ['ERR_MAG_6_gSDSS',     'ERR_MAG_6_rSDSS',     'ERR_MAG_6_iSDSS',      'ERR_MAG_6_J0395',     'ERR_MAG_6_J0410',     'ERR_MAG_6_J0430',     'ERR_MAG_6_J0515',    'ERR_MAG_6_J0660',    'ERR_MAG_6_J0861']

}
###########################################################################
