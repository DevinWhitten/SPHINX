#########################################################################
### This is the main parameter file
{
"mag_err_max" : 0.25,
"mag_faint_lim" : 20,
"mag_bright_lim" : 14,

"EBV_MAX": 0.2,
"scale_frame": "self",
"band_type": "native",
"solver": "adam",

"SPHINX_path"    : "/scratch/SPHINX/",
"target_path"    : "datasets/SPLUS_STRIPE82_tile_ebv_cor.csv.gz",

'TEFF_NET'       : 'TEFF_NET.pkl.gz',

'FEH_NET_COOL'   : 'FEH_NET_4750_5750.pkl.gz',
'FEH_NET_WARM'    : 'FEH_NET_5500_7000.pkl.gz',

'AC_NET_COOL'   : 'FEH_NET_4750_5750.pkl.gz',
'AC_NET_WARM'    : 'FEH_NET_5500_7000.pkl.gz',


"output_directory": "output/SPLUS82/final/",
"output_filename":"SPLUS_full_cor_4250_5750.csv",

'target_bands': ["gSDSS_c",  'rSDSS_c',  'iSDSS_c',  'F395_c', 'F410_c', 'F430_c', 'F515_c', 'F660_c', 'F861_c'],
"target_sigma": ["eg_aper", 'er_aper', 'ei_aper', 'eF395_aper', 'eF410_aper', 'eF430_aper', 'eF515_aper', 'eF660_aper', 'eF861_aper'],

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
#"target_sigma": ['ERR_MAG_6_gSDSS', 'ERR_MAG_6_rSDSS', 'ERR_MAG_6_iSDSS', 'ERR_MAG_6_J0395', "ERR_MAG_6_J0410", 'ERR_MAG_6_J0430', 'ERR_MAG_6_J0515','ERR_MAG_6_J0660','ERR_MAG_6_J0861'],
