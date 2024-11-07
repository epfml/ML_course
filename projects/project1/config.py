dictionary_features = {
    **dict.fromkeys(['_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR',
                     'DISPCODE', 'SEQNO', '_PSU', 'SEX', 'QSTVER', '_STSTR',
                     '_STRWT', '_RAWRAKE', '_WT2RAKE', '_DUALUSE', '_LLCPWT',
                     '_DRDXAR1', '_RACE_G1', '_AGE80', '_AGE_G', 'HTIN4', 'HTM4', '_BMI5', '_BMI5CAT',
                     'FTJUDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_MISFRTN', '_MISVEGN',
                     '_FRTRESP', '_VEGRESP', '_FRUTSUM', '_FRUTSUM', '_FRT16', '_VEG23', '_FRUITEX',
                     '_VEGETEX', 'QSTLANG', 'FRUTDA1_', '_VEGESUM'
                   
                    ], {}),
    
    **dict.fromkeys(['ALCDAY5', 'STRENGTH'], {'dont_know_not_sure' : 777, 'none' : 888, 'refused' : 999}),
    
    **dict.fromkeys(['PHYSHLTH', 'MENTHLTH'], {'dont_know_not_sure' : 77, 'none' : 88, 'refused' : 99}),
    
    **dict.fromkeys(['CHILDREN'], {'none' : 88, 'refused' : 99}),
    
     **dict.fromkeys(['INCOME2', '_PRACE1', '_MRACE1'], {'dont_know_not_sure' : 77, 'refused' : 99}),

    **dict.fromkeys(['FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVORANG', 'FVGREEN', 'VEGETAB1'], {'never' : 555, 'dont_know_not_sure' : 777, 'refused' : 999}),
    
     **dict.fromkeys(['GENHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'BPHIGH4',
                      'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3',
                      'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2',
                      'CHCKIDNY', 'DIABETE3', 'RENTHOM1', 'VETERAN3', 'INTERNET',
                     'QLACTLM2', 'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES',
                      'DIFFALON', 'SMOKE100', 'USENOW3', 'EXERANY2', 'FLUSHOT6', 'PNEUVAC3',
                      'HIVTST6', 'DRNKANY5'
                     ], {'dont_know_not_sure' : 7, 'refused' : 9}),
    
    **dict.fromkeys(['CHECKUP1', 'SEATBELT'], {'dont_know_not_sure' : 7, 'never' : 8, 'refused' : 9}),
    
    **dict.fromkeys(['MARITAL', 'EDUCA', 'EMPLOY1', '_CHISPNC', '_RFHLTH',
                      '_HCVU651', '_RFHYPE5', '_CHOLCHK', '_RFCHOL', '_LTASTH1',
                      '_CASTHM1', '_ASTHMS1', '_HISPANC', '_RACE', '_RACEG21',
                      '_RACEGR3', '_RFBMI5', '_CHLDCNT', '_EDUCAG', '_INCOMG',
                      '_SMOKER3', '_RFSMOK3', '_RFBING5', '_RFDRHV5', '_FRTLT1',
                     '_VEGLT1', '_TOTINDA', 'PAMISS1_', '_PACAT1', '_PAINDX1', '_PA150R2',
                     '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_PASTAE1', '_LMTACT1',
                     '_LMTWRK1', '_LMTSCL1', '_RFSEAT2', '_RFSEAT3', '_AIDTST3'
                     ], {'refused' : 9}),
    
    **dict.fromkeys(['HEIGHT3', 'WEIGHT2'], {'dont_know_not_sure' : 7777, 'refused' : 9999}),
    
    **dict.fromkeys(['_AGEG5YR'], {'refused' : 14}),

    **dict.fromkeys(['_AGE65YR'], {'refused' : 3}),

    **dict.fromkeys(['WTKG3'], {'refused' : 99999}),
    **dict.fromkeys(['DROCDY3_'], {'refused' : 900}),
    **dict.fromkeys(['_DRNKWEK'], {'refused' : 99900}),
     **dict.fromkeys(['FC60_', 'MAXVO2_'], {'refused' : 999}),
     **dict.fromkeys(['STRFREQ_'], {'refused' : 99}),
   
}

category_features = {'categorical' : ['HLTHPLN1','_STATE','FMONTH','IDATE', 'IMONTH', 'DISPCODE', 'PERSDOC2',
                             'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3',
                            'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY'
                             'DIABETE3','SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 'EMPLOY1', 'INCOME2', 'INTERNET', 'QLACTLM2',
                             'USEEQUIP', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'USENOW3', 'EXERANY2', 'SEATBELT',
                             'FLUSHOT6', 'PNEUVAC3', 'HIVTST6', 'QSTVER', 'QSTLANG','_CHISPNC', '_DUALUSE', '_RFHLTH', '_HCVU651', '_RFHYPE5',
                             '_CHOLCHK', '_RFCHOL', '_LTASTH1', '_CASTHM1', '_ASTHMS1', '_DRDXAR1', '_PRACE1', '_MRACE1', '_HISPANC', '_RACE',
                             '_RACEG21', '_RACEGR3',  '_RACE_G1', '_AGEG5YR', '_AGE65YR', '_AGE_G', '_BMI5CAT', '_RFBMI5', '_CHLDCNT', '_EDUCAG',
                             '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', '_RFBING5', '_RFDRHV5', '_MISFRTN', '_MISVEGN', '_FRTRESP', '_VEGRESP',
                             '_FRTLT1', '_VEGLT1', '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX', '_TOTINDA', 'PAMISS1_', '_PACAT1', '_PAINDX1',
                             '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_PASTAE1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1',
                             '_RFSEAT2', '_RFSEAT3', '_AIDTST3', 'IYEAR', '_RFSMOK3', 'FMONTH', 'GENHLTH', 'CHCKIDNY', 'DIABETE3'
                             
                             
                            ],

            
            'continuous' : ['WEIGHT2', 'HEIGHT3', 'ALCDAY5', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN',
                            'FVORANG', 'VEGETAB1', 'STRENGTH', '_AGE80', 'HTIN4', 'WTKG3', 'HTM4', '_BMI5', 'DROCDY3_',
                           '_DRNKWEK', 'FTJUDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_', '_FRUTSUM', '_FRUTSUM',
                           'MAXVO2_', 'FC60_', 'STRFREQ_', 'PHYSHLTH', 'IDAY', 'CHILDREN', 'MENTHLTH', 'FRUTDA1_', '_STRWT', '_VEGESUM'],
            
            'not_sure' : ['SEQNO', '_PSU' , '_STSTR', '_STRWT', '_RAWRAKE', ' _WT2RAKE', '_LLCPWT']
           }