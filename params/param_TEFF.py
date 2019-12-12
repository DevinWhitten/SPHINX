#########################################################################
{
"TEFF_MIN" : 4000,
"TEFF_MAX" : 7500,
"TEFF_ERR_MAX" : 120.,

"FEH_MAX" : -0.5,
"FEH_MIN" : -3.2,
"FEH_ERR_MAX": 0.20,


"CFE_MIN" : -1.0,
"CFE_MAX" : 3.0,
"CFE_ERR_MAX": 0.20,

"AC_MIN" : 5.0,
"AC_MAX" : 8.5,
"AC_ERR_MAX": 0.20,

"mag_err_max" : 0.10,
"mag_faint_lim" : 19,
"mag_bright_lim" : 14,

"EBV_MAX": 0.1,
"scale_frame": "self",
"band_type": "native",
'input_number': 6,
'array_size': 500,
'hidden_layers': (8, 8),
'skim' : 5,
"train_iterations": 2,
"solver": "adam",

"SPHINX_path"    : "/emc3/SPHINX/",
"target_path"    : "datasets/a_-1.00_l_-1.75.csv",
"training_path"  : "datasets/full_catalog_201911.csv.gz",
'segue_path'     : 'datasets/full_catalog_201911.csv.gz',


"output_directory": "output/SPLUS82/",
"output_filename":"a_-1.00_l_-1.75.csv",

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
#"target_bands": ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',     'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861']
#"target_bands": ['gMag',     'rMag',     'iMag',     'F395Mag',     'F410Mag',     'F430Mag',     'F515Mag',    'F660Mag',    'F861Mag'],,
#"idr_segue_bands":    ['MAG_6_gSDSS',     'MAG_6_rSDSS',     'MAG_6_iSDSS',      'MAG_6_J0395',     'MAG_6_J0410',     'MAG_6_J0430',     'MAG_6_J0515',    'MAG_6_J0660',    'MAG_6_J0861'],
#"target_sigma"<F4>: ['ERR_MAG_6_gSDSS', 'ERR_MAG_6_rSDSS', 'ERR_MAG_6_iSDSS', 'ERR_MAG_6_J0395', "ERR_MAG_6_J0410", 'ERR_MAG_6_J0430', 'ERR_MAG_6_J0515','ERR_MAG_6_J0660','ERR_MAG_6_J0861'],
