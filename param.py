#########################################################################
params = {
"TMIN" : 5500,
"TMAX" : 8000,
"T_ERR_MAX" : 120.,

"FEH_MAX" : 0.0,
"FEH_MIN" : -4.0,
"FEH_ERR_MAX": 0.20,

"CFE_MAX" : 0.00,

"mag_err_max" : 0.10,
"mag_faint_lim" : 18,
"mag_bright_lim" : 14,

"EBV_MAX": 0.05,
"band_type":"synthetic",
"target_var": "FEH_BIW",
"format_var": "FEH",
'array_size': 5,
"target_path" : "datasets/IDR_201803_testing_native_feh.csv",
"segue_path"  : "datasets/SEGUE_calibrated_catalog_GOLD.csv.gz",
"edr_segue_path": "datasets/EDR_SEGUE.csv",
"idr_segue_path": "datasets/IDR_201803_training_sup_feh.csv",
"idr_segue_sup_path": "datasets/IDR_SEGUE_VMP_sup.csv",
"idr_segue_dr10_path": "datasets/IDR_DR10_SEGUE_SUPPLEMENTED.csv",

"output_directory": "output/",
"output_filename":"example_output_file.csv",
'target_bands': ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',     'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861'],
"target_sigma": ['ERR_MAG_6_gSDSS', 'ERR_MAG_6_rSDSS', 'ERR_MAG_6_iSDSS', 'ERR_MAG_6_J0395', "ERR_MAG_6_J0410", 'ERR_MAG_6_J0430', 'ERR_MAG_6_J0515','ERR_MAG_6_J0660','ERR_MAG_6_J0861'],

"format_bands": ['gSDSS', 'rSDSS', 'iSDSS',  'F395',  'F410',  'F430',  'F515',  'F660',  'F861'],

"segue_bands":  ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"segue_sigma":  ["gMag_Sigma", 'rMag_Sigma', 'iMag_Sigma', 'F395Mag_Sigma', 'F410Mag_Sigma', 'F430Mag_Sigma', 'F515Mag_Sigma', 'F660Mag_Sigma', 'F861Mag_Sigma'],

"idr_segue_bands": ['gMag',     'rMag',     'iMag',     'F395Mag',     'F410Mag',     'F430Mag',     'F515Mag',    'F660Mag',    'F861Mag'],
"idr_segue_sigma": ['ERR_MAG_6_gSDSS',     'ERR_MAG_6_rSDSS',     'ERR_MAG_6_iSDSS',      'ERR_MAG_6_J0395',     'ERR_MAG_6_J0410',     'ERR_MAG_6_J0430',     'ERR_MAG_6_J0515',    'ERR_MAG_6_J0660',    'ERR_MAG_6_J0861']

}
###########################################################################
#"target_bands": ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',     'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861']
#"target_bands": ['gMag',     'rMag',     'iMag',     'F395Mag',     'F410Mag',     'F430Mag',     'F515Mag',    'F660Mag',    'F861Mag'],,
#"idr_segue_bands":    ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',      'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861'],
