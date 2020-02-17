import pandas as pd
import numpy as np
import sys

def merge_datasets(COOL, HOT, TCRIT = [5500, 5750]):
    print("... merging datasets")
    print("\t presizes:  LEFT:  ", len(COOL), "  RIGHT:    ", len(HOT))
    print("\t Teff boundary for join:  ", TCRIT)

    ## Tries to merge the FEH/AC estimate in the transition region
    LEFT = COOL[COOL['NET_TEFF'] < TCRIT[0]].copy()
    RIGHT = HOT[HOT['NET_TEFF'] > TCRIT[1]].copy()

    MID_LEFT = COOL[COOL['NET_TEFF'].between(*TCRIT, inclusive=True)].copy()
    MID_RIGHT = HOT[HOT['NET_TEFF'].between(*TCRIT, inclusive=True)].copy()

    ### want to join the mid, while maximizing stars with estimates

    ID = pd.merge(MID_LEFT, MID_RIGHT, on='ID_UNQ')['ID_UNQ']

    ### save the observations unique to each side

    MID_LEFT_save = MID_LEFT[~MID_LEFT['ID_UNQ'].isin(ID)].copy()
    MID_RIGHT_save = MID_RIGHT[~MID_RIGHT['ID_UNQ'].isin(ID)].copy()

    ### now prepare the common stars
    MID_LEFT_COMMON = MID_LEFT[MID_LEFT['ID_UNQ'].isin(ID)].copy()
    MID_RIGHT_COMMON = MID_RIGHT[MID_RIGHT['ID_UNQ'].isin(ID)].copy()

    print(len(MID_LEFT_COMMON), len(MID_RIGHT_COMMON))

    ### Rename columns

    MID_LEFT_COMMON = MID_LEFT_COMMON.rename(columns={"NET_FEH": "NET_FEH_LEFT", "NET_FEH_ERR": "NET_FEH_ERR_LEFT",
                                                      "NET_AC": "NET_AC_LEFT", "NET_AC_ERR": "NET_AC_ERR_LEFT",
                                                      "NET_ARRAY_FEH_FLAG": 'NET_ARRAY_FEH_FLAG_LEFT',
                                                      "NET_ARRAY_AC_FLAG" : 'NET_ARRAY_AC_FLAG_LEFT'},
                                                      errors="raise")

    MID_RIGHT_COMMON = MID_RIGHT_COMMON.rename(columns={"NET_FEH": "NET_FEH_RIGHT", "NET_FEH_ERR": "NET_FEH_ERR_RIGHT",
                                                        "NET_AC": "NET_AC_RIGHT", "NET_AC_ERR": "NET_AC_ERR_RIGHT",
                                                        "NET_ARRAY_FEH_FLAG": 'NET_ARRAY_FEH_FLAG_RIGHT',
                                                        "NET_ARRAY_AC_FLAG" : 'NET_ARRAY_AC_FLAG_RIGHT'},
                                                        errors="raise")

    ### Now merge the estimates
    COMBO = pd.merge(MID_LEFT_COMMON[['NET_FEH_LEFT', 'NET_FEH_ERR_LEFT', 'NET_AC_LEFT', 'NET_AC_ERR_LEFT',
                                      'NET_ARRAY_FEH_FLAG_LEFT', 'NET_ARRAY_AC_FLAG_LEFT', 'ID_UNQ']],
                     MID_RIGHT_COMMON[['NET_FEH_RIGHT', 'NET_FEH_ERR_RIGHT', 'NET_AC_RIGHT', 'NET_AC_ERR_RIGHT',
                                      'NET_ARRAY_FEH_FLAG_RIGHT', 'NET_ARRAY_AC_FLAG_RIGHT', 'ID_UNQ']],
                                      on='ID_UNQ')
    print('COMBO:   ', len(COMBO))
    FEH_FLAG = []
    FEH_VALUE = []
    FEH_ERR_VALUE = []
    for i, row in COMBO.iterrows():
        if np.isfinite(row['NET_FEH_LEFT']) & np.isfinite(row['NET_FEH_RIGHT']) & \
           np.isfinite(row['NET_FEH_ERR_LEFT']) & np.isfinite(row['NET_FEH_ERR_RIGHT']) & \
           (row['NET_ARRAY_FEH_FLAG_LEFT'] > 1) & (row['NET_ARRAY_FEH_FLAG_RIGHT'] > 1):
        ### then do a nice weighted average

            #FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']],
            #                        weights=[1./row['NET_FEH_ERR_LEFT'], 1./row['NET_FEH_ERR_RIGHT']]))

            FEH_ERR_VALUE.append(max([row['NET_FEH_ERR_LEFT'], row['NET_FEH_ERR_RIGHT']]))
            FEH_FLAG.append(min([row['NET_ARRAY_FEH_FLAG_LEFT'], row['NET_ARRAY_FEH_FLAG_RIGHT']]))


        ### then do a nice weighted average
            try:
                FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']],
                                    weights=[1./row['NET_FEH_ERR_LEFT'], 1./row['NET_FEH_ERR_RIGHT']]))
            except:
                print("weird", row['NET_FEH_ERR_LEFT'], row['NET_ARRAY_FEH_FLAG_LEFT'], row['NET_FEH_ERR_RIGHT'], row['NET_ARRAY_FEH_FLAG_RIGHT'])
                FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']]))




        ###  then take the one that works
        elif np.isfinite(row['NET_FEH_LEFT']) & np.isfinite(row['NET_FEH_ERR_LEFT']) & (row['NET_ARRAY_FEH_FLAG_LEFT'] > 1):
            FEH_VALUE.append(row['NET_FEH_LEFT'])
            FEH_ERR_VALUE.append(row['NET_FEH_ERR_LEFT'])
            FEH_FLAG.append(row['NET_ARRAY_FEH_FLAG_LEFT'])



        elif np.isfinite(row['NET_FEH_RIGHT']) & np.isfinite(row['NET_FEH_ERR_RIGHT']) & (row['NET_ARRAY_FEH_FLAG_RIGHT'] > 1):
            FEH_VALUE.append(row['NET_FEH_RIGHT'])
            FEH_ERR_VALUE.append(row['NET_FEH_ERR_RIGHT'])
            FEH_FLAG.append(row['NET_ARRAY_FEH_FLAG_RIGHT'])


        else:
            print("Bad row: ", i, end='\r')
            sys.stdout.flush()
            FEH_VALUE.append(np.nan)
            FEH_ERR_VALUE.append(np.nan)
            FEH_FLAG.append(np.nan)


    ### Now let's just repeat with AC
    AC_FLAG = []
    AC_VALUE = []
    AC_ERR_VALUE = []
    for i, row in COMBO.iterrows():
        if np.isfinite(row['NET_AC_LEFT']) & np.isfinite(row['NET_AC_RIGHT']) & \
           np.isfinite(row['NET_AC_ERR_LEFT']) & np.isfinite(row['NET_AC_ERR_RIGHT']) & \
           (row['NET_ARRAY_AC_FLAG_LEFT'] > 1) & (row['NET_ARRAY_AC_FLAG_RIGHT'] > 1):
        ### then do a nice weighted average
            try:
                AC_VALUE.append(np.average([row['NET_AC_LEFT'], row['NET_AC_RIGHT']],
                                    weights=[1./row['NET_AC_ERR_LEFT'], 1./row['NET_AC_ERR_RIGHT']]))
            except:
                print("weird", row['NET_AC_ERR_LEFT'], row['NET_AC_ERR_RIGHT'])
                AC_VALUE.append(np.average([row['NET_AC_LEFT'], row['NET_AC_RIGHT']]))


            AC_ERR_VALUE.append(max([row['NET_AC_ERR_LEFT'], row['NET_AC_ERR_RIGHT']]))
            AC_FLAG.append(min([row['NET_ARRAY_AC_FLAG_LEFT'], row['NET_ARRAY_AC_FLAG_RIGHT']]))

        ###  then take the one that works
        elif np.isfinite(row['NET_AC_LEFT']) & np.isfinite(row['NET_AC_ERR_LEFT']) & (row['NET_ARRAY_AC_FLAG_LEFT'] > 1):
            AC_VALUE.append(row['NET_AC_LEFT'])
            AC_ERR_VALUE.append(row['NET_AC_ERR_LEFT'])
            AC_FLAG.append(row['NET_ARRAY_AC_FLAG_LEFT'])


        elif np.isfinite(row['NET_AC_RIGHT']) & np.isfinite(row['NET_AC_ERR_RIGHT']) & (row['NET_ARRAY_AC_FLAG_RIGHT'] > 1):
            AC_VALUE.append(row['NET_AC_RIGHT'])
            AC_ERR_VALUE.append(row['NET_AC_ERR_RIGHT'])
            AC_FLAG.append(row['NET_ARRAY_AC_FLAG_RIGHT'])


        else:
            print("Bad:  ", i, end='\r')
            sys.stdout.flush()
            AC_VALUE.append(np.nan)
            AC_ERR_VALUE.append(np.nan)
            AC_FLAG.append(np.nan)



    COMBO.loc[:, 'NET_FEH']            = FEH_VALUE
    COMBO.loc[:, 'NET_FEH_ERR']        = FEH_ERR_VALUE
    COMBO.loc[:, 'NET_ARRAY_FEH_FLAG'] = FEH_FLAG

    COMBO.loc[:, 'NET_AC']            = AC_VALUE
    COMBO.loc[:, 'NET_AC_ERR']        = AC_ERR_VALUE
    COMBO.loc[:, 'NET_ARRAY_AC_FLAG'] = AC_FLAG


    for label in ['NET_FEH_LEFT', 'NET_FEH_RIGHT', 'NET_FEH_ERR_LEFT', 'NET_FEH_ERR_RIGHT',
                  'NET_AC_LEFT', 'NET_AC_RIGHT', 'NET_AC_ERR_LEFT', 'NET_AC_ERR_RIGHT',
                  'NET_ARRAY_FEH_FLAG_LEFT', 'NET_ARRAY_FEH_FLAG_RIGHT', 'NET_ARRAY_AC_FLAG_LEFT', 'NET_ARRAY_AC_FLAG_RIGHT']:
        del COMBO[label]

    for frame in [LEFT, RIGHT, MID_LEFT_save, MID_RIGHT_save]:
        frame.loc[:, 'fix_feh'] = np.zeros(len(frame))
        frame.loc[:, 'fix_ac'] = np.zeros(len(frame))


    COMBO.loc[:, 'fix_feh'] = np.ones(len(COMBO))
    COMBO.loc[:, 'fix_ac'] = np.ones(len(COMBO))

    print('COMBO', len(COMBO))

    COMBO = pd.merge(COMBO, MID_LEFT_COMMON, on='ID_UNQ')

    return pd.concat([LEFT, RIGHT, MID_LEFT_save, MID_RIGHT_save, COMBO], sort=True)
