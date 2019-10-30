#########################################################################
params = {
"TMIN" : 4500,
"TMAX" : 8000,
"T_ERR_MAX" : 120.,

"FEH_MAX" : 0.0,
"FEH_MIN" : -4.0,
"FEH_ERR_MAX": 0.20,


"CFE_MIN" : -1.0,
"CFE_MAX" : 3.0,
"CFE_ERR_MAX": 0.20,

"AC_MIN" : 5.0,
"AC_MAX" : 8.5,

"mag_err_max" : 0.10,
"mag_faint_lim" : 20,
"mag_bright_lim" : 14,

"EBV_MAX": 0.1,
"scale_frame": "self",
"band_type": "native",
'input_number': 6,
'array_size': 10,
"train_iterations": 2,
"solver": "adam",


"target_path"    : "datasets/SPLUS82_test.csv",
"training_path"  : "datasets/SPLUS82_train.csv",
'segue_path'     : 'datasets/SEGUE_DR10_Cal_Catalog_SPLUS82_Mag_v2.csv.gz',


"output_directory": "output/SPLUS82/",
"output_filename":"first.csv",

'target_bands': ["g_cor",  'r_cor',  'i_cor',  'F395_cor', 'F410_cor', 'F430_cor', 'F515_cor', 'F660_cor', 'F861_cor'],
"target_sigma": ["eg_aper", 'er_aper', 'ei_aper', 'eF395_aper', 'eF410_aper', 'eF430_aper', 'eF515_aper', 'eF660_aper', 'eF861_aper'],

"format_bands": ['gSDSS', 'rSDSS', 'iSDSS',  'F395',  'F410',  'F430',  'F515',  'F660',  'F861'],

"segue_bands":  ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"segue_sigma":  ["gMag_Sigma", 'rMag_Sigma', 'iMag_Sigma', 'F395Mag_Sigma', 'F410Mag_Sigma', 'F430Mag_Sigma', 'F515Mag_Sigma', 'F660Mag_Sigma', 'F861Mag_Sigma'],

'training_bands': ["g_cor",  'r_cor',  'i_cor',  'F395_cor', 'F410_cor', 'F430_cor', 'F515_cor', 'F660_cor', 'F861_cor'],
'training_sigma': ["eg_aper", 'er_aper', 'ei_aper', 'eF395_aper', 'eF410_aper', 'eF430_aper', 'eF515_aper', 'eF660_aper', 'eF861_aper'],

"synth_bands": ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"native_bands":['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',     'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861'],

'training_var':     {'TEFF': 'TEFFA', 'FEH': 'FEHF',
                     'CFE': 'CARF', 'AC': 'AC'},

'training_var_err' :{'TEFF_ERR' : 'TEFFA_ERR', 'FEH_ERR' : 'FEHF_ERR',
                     'CFE_ERR' : 'CARF_ERR', 'AC_ERR' : 'AC_ERR'},

'segue_var'  : {'TEFF' : 'TEFF_ADOP', 'FEH' : 'FEH_BIW', 'CFE' : 'CFE_COR', 'AC' : 'AC'},
'segue_var_err' : {'TEFF_ERR' : "TEFF_ADOP_ERR", 'FEH_ERR' : 'FEH_BIW_ERR', 'CFE_ERR' : 'CFE_ERR', 'AC_ERR' : 'AC_ERR'}
}
###########################################################################
#"target_bands": ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',     'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861']
#"target_bands": ['gMag',     'rMag',     'iMag',     'F395Mag',     'F410Mag',     'F430Mag',     'F515Mag',    'F660Mag',    'F861Mag'],,
#"idr_segue_bands":    ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',      'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861'],
#"target_sigma": ['ERR_MAG_6_gSDSS', 'ERR_MAG_6_rSDSS', 'ERR_MAG_6_iSDSS', 'ERR_MAG_6_J0395', "ERR_MAG_6_J0410", 'ERR_MAG_6_J0430', 'ERR_MAG_6_J0515','ERR_MAG_6_J0660','ERR_MAG_6_J0861'],
