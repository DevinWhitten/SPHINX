#########################################################################
{
"TEFF_MIN" : 5500,
"TEFF_MAX" : 7000,
"T_ERR_MAX" : 120.,

"FEH_MAX" : 0.0,
"FEH_MIN" : -3.2,
"FEH_ERR_MAX": 0.20,


"CFE_MIN" : -1.0,
"CFE_MAX" : 3.0,
"CFE_ERR_MAX": 0.20,

"AC_MIN" : 5.0,
"AC_MAX" : 8.5,
"AC_ERR_MAX": 0.25,

"mag_err_max" : 0.10,
"mag_faint_lim" : 19.5,
"mag_bright_lim" : 9,

"EBV_MAX": 0.1,
"scale_frame": "self",
"band_type": "native",
'input_number': 6,
'array_size': 750,
'hidden_layers': (10, 10),
'skim' : 8,
"train_iterations": 2,
"solver": "adam",

"SPHINX_path"    : "~/Google Drive/SPHINX/",
"training_path"  : "datasets/full_catalog_202002.csv.gz",
'segue_path'     : 'datasets/full_catalog_202002.csv.gz',


"output_directory": "output/SPLUS82/",

'target_bands': ["gSDSS",  'rSDSS',  'iSDSS',  'F395', 'F410', 'F430', 'F515', 'F660', 'F861'],
"target_sigma": ["gSDSS_err", 'rSDSS_err', 'iSDSS_err', 'F395_err', 'F410_err', 'F430_err', 'F515_err', 'F660_err', 'F861_err'],

"format_bands": ['gSDSS', 'rSDSS', 'iSDSS',  'F395',  'F410',  'F430',  'F515',  'F660',  'F861'],

"segue_bands":  ["gMag",  'rMag',  'iMag',  'F395Mag', 'F410Mag', 'F430Mag', 'F515Mag', 'F660Mag', 'F861Mag'],
"segue_sigma":  ["gMag_Sigma", 'rMag_Sigma', 'iMag_Sigma', 'F395Mag_Sigma', 'F410Mag_Sigma', 'F430Mag_Sigma', 'F515Mag_Sigma', 'F660Mag_Sigma', 'F861Mag_Sigma'],

'training_bands': ["gSDSS",  'rSDSS',  'iSDSS',  'F395', 'F410', 'F430', 'F515', 'F660', 'F861'],
'training_sigma': ["gSDSS_err", 'rSDSS_err', 'iSDSS_err', 'F395_err', 'F410_err', 'F430_err', 'F515_err', 'F660_err', 'F861_err'],

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
