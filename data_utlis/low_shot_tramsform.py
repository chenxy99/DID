def VOC_low_shot_dataset_transform(dataset, num_shot, num_classes):

    used_index = []
    dataset_len = len(dataset.ids)
    counter = 0
    for target_index in range(1, num_classes):
        finished_num = 0
        while 1:
            if (counter % dataset_len) not in used_index:
                img, target = dataset.__getitem__(counter % dataset_len)
                if (target[:, -1] == target_index).sum() > 0:
                    #print(len(used_index))
                    used_index.append(counter % dataset_len)
                    finished_num += 1
                    if finished_num == num_shot:
                        break
            counter += 1

    import numpy as np
    
    record = np.zeros((len(dataset.ids), num_classes))
    for index in range(len(dataset.ids)):
        img, target = dataset.__getitem__(index)
        for j in range(1, num_classes):
            record[index, j] = (target[:, -1] == j).sum()

    for index in range(len(dataset.ids)):
        record_index = record[index]
        if (record_index > 0).sum() >= 4:
            record[index] = 0
        if (record_index >= 4).sum() >= 1:
            record[index] = 0

    print('Preprocess finish')

    used_index = []
    dataset_len = len(dataset.ids)
    counter = 104
    for target_index in range(1, num_classes):
        finished_num = 0
        while 1:
            if (counter % dataset_len) not in used_index:
                if record[counter % dataset_len, target_index] > 0.01:
                    print(len(used_index))
                    used_index.append(counter % dataset_len)
                    finished_num += 1
                    if finished_num == num_shot:
                        break
            counter += 1
            if counter % dataset_len == 0:
                print('one circle')

    a = 1

    ''''''


    '''
    #tpami 10
    used_index = [13, 211, 233, 489, 776, 1217, 1523, 1696, 1705, 1745, 1752, 1753, 1820, 2083, 2192, 2193, 2223, 2262,
                  2310, 2317, 2692, 2882, 2975, 3061, 3265, 3381, 3398, 3425, 3893, 3946, 4075, 4156, 4248, 4455, 4828,
                  5003, 29, 35, 61, 75, 113, 127, 174, 189, 243, 264, 277, 350, 464, 476, 554, 570, 685, 709, 716, 792,
                  934, 970, 999, 1001, 1009, 1015, 1046, 1050, 1060, 1068, 1073, 1075, 1077, 1088, 1091, 1126, 1148,
                  1187, 1195, 1202, 1216, 1218, 1222, 1231, 1248, 1254, 1283, 1284, 1325, 1353, 1356, 1379, 1381, 1390,
                  1403, 1495, 1501, 1560, 1687, 1719, 1736, 1833, 1889, 2015, 2217, 2252, 2388, 2438, 2535, 2550, 2551,
                  2624, 2676, 2771, 2773, 2778, 2810, 2817, 2856, 2858, 2885, 2888, 2900, 2908, 2956, 2989, 3058, 3079,
                  3088, 3146, 3162, 3168, 3173, 3216, 3243, 3266, 3290, 3315, 3331, 3347, 3400, 3408, 3436, 3446, 3447,
                  3452, 3455, 3457, 3464, 3469, 3481, 3485, 3493, 3496, 3511, 3575, 3602, 3691, 3710, 3789, 3828, 3838,
                  3867, 3885, 3960, 4138, 4204, 4446, 4470, 4629, 4649, 4877, 4951, 48, 67, 76, 96, 134, 231, 247, 253,
                  293, 307, 318, 324, 333, 345, 377, 384, 401, 410, 427, 447, 460, 462, 473, 491, 495, 502, 503, 509,
                  524, 533, 568]
    '''

    '''
    # tpami 3
    used_index = [13, 14, 53, 59, 60, 151, 172, 181, 194, 230, 250, 255, 264, 277, 350, 361, 365, 373, 378, 383, 401,
                  423, 434, 455, 456, 457, 458, 558, 563, 585, 627, 704, 711, 727, 740, 754, 755, 757, 775, 836, 874,
                  882, 884, 885, 887, 937, 993, 1011, 1093, 1122, 1125, 1126, 1160, 1163, 1172, 1210, 1232, 1261, 1279,
                  1284]
    '''

    '''
    # 20 classes 10 shot
    used_index = [13, 14, 53, 115, 129, 135, 163, 211, 233, 251, 252, 265, 280, 310, 322, 334, 365, 366, 374, 385, 387,
                  411, 419, 437, 457, 477, 480, 488, 499, 506, 553, 559, 584, 598, 620, 648, 660, 723, 734, 736, 740,
                  756, 765, 779, 785, 789, 825, 868, 880, 913, 934, 940, 970, 999, 1001, 1090, 1096, 1104, 1117, 1132,
                  1135, 1141, 1147, 1150, 1165, 1167, 1174, 1185, 1197, 1204, 1216, 1218, 1222, 1231, 1236, 1243, 1245,
                  1322, 1342, 1358, 1379, 1381, 1390, 1395, 1406, 1408, 1409, 1436, 1455, 1471, 1472, 1495, 1501, 1560,
                  1687, 1719, 1736, 1833, 1889, 1961, 1977, 2043, 2090, 2136, 2152, 2163, 2217, 2220, 2248, 2252, 2264,
                  2282, 2284, 2291, 2301, 2305, 2313, 2325, 2329, 2347, 2351, 2370, 2396, 2397, 2410, 2420, 2440, 2453,
                  2466, 2473, 2485, 2553, 2564, 2582, 2589, 2608, 2610, 2630, 2632, 2660, 2661, 2666, 2667, 2669, 2670,
                  2674, 2675, 2677, 2678, 2681, 2767, 2777, 2812, 2831, 2849, 2852, 2855, 2894, 2911, 2919, 2988, 2997,
                  3074, 3081, 3145, 3148, 3291, 3364, 3365, 3442, 3454, 3456, 3470, 3491, 3494, 3515, 3518, 3553, 3565,
                  3575, 3588, 3598, 3603, 3624, 3651, 3653, 3676, 3712, 3731, 3753, 3760, 3763, 3785, 3797, 3805, 3884,
                  3897, 3899, 3911, 3914]
    '''
    '''
    # 20 classes 30 shot
    used_index = [13, 14, 53, 115, 129, 135, 163, 211, 233, 251, 274, 298, 312, 317, 353, 356, 370, 399, 415, 432, 440,
                  445, 453, 466, 468, 489, 504, 544, 547, 564, 586, 588, 592, 605, 607, 625, 650, 651, 694, 720, 722,
                  747, 766, 772, 784, 795, 805, 808, 811, 821, 853, 906, 922, 941, 1019, 1022, 1069, 1115, 1118, 1127,
                  1130, 1149, 1161, 1199, 1235, 1278, 1291, 1294, 1297, 1306, 1314, 1320, 1335, 1359, 1364, 1365, 1366,
                  1386, 1402, 1423, 1451, 1462, 1470, 1491, 1496, 1541, 1542, 1543, 1546, 1565, 1625, 1679, 1715, 1721,
                  1832, 1843, 1850, 1880, 1914, 1921, 1946, 1947, 1948, 1975, 2004, 2109, 2131, 2134, 2140, 2176, 2202,
                  2209, 2240, 2256, 2275, 2304, 2333, 2423, 2446, 2465, 2470, 2529, 2566, 2584, 2601, 2622, 2624, 2653,
                  2688, 2700, 2702, 2727, 2746, 2749, 2781, 2790, 2920, 2973, 3042, 3084, 3105, 3112, 3143, 3175, 3191,
                  3207, 3215, 3223, 3226, 3249, 3277, 3307, 3405, 3410, 3435, 3530, 3531, 3572, 3585, 3609, 3657, 3658,
                  3673, 3677, 3688, 3707, 3721, 3730, 3749, 3776, 3806, 3819, 3845, 3852, 3901, 3920, 3957, 3984, 3992,
                  4032, 4038, 4043, 4047, 4051, 4056, 4057, 4068, 4074, 4076, 4083, 4086, 4107, 4112, 4115, 4120, 4124,
                  4140, 4143, 4159, 4171, 4173, 4195, 4198, 4207, 4208, 4215, 4221, 4223, 4224, 4238, 4244, 4251, 4268,
                  4284, 4291, 4292, 4296, 4370, 4392, 4402, 4416, 4421, 4439, 4440, 4453, 4456, 4468, 4476, 4479, 4503,
                  4533, 4539, 4557, 4561, 4583, 4599, 4627, 4635, 4657, 4668, 4683, 4687, 4689, 4697, 4703, 4724, 4729,
                  4737, 4745, 4756, 4785, 4788, 4796, 4797, 4798, 4802, 4815, 4832, 4835, 4846, 4849, 4850, 4856, 4886,
                  4890, 4897, 4902, 4909, 4916, 4925, 4962, 4980, 4986, 48, 105, 117, 152, 201, 204, 224, 226, 239, 303,
                  351, 372, 374, 382, 395, 406, 412, 451, 531, 558, 563, 585, 649, 654, 713, 728, 802, 810, 828, 832,
                  880, 888, 907, 932, 939, 971, 1000, 1002, 1017, 1018, 1038, 1062, 1101, 1181, 1193, 1200, 1219, 1227,
                  1268, 1274, 1316, 1321, 1332, 1334, 1338, 1382, 1406, 1412, 1418, 1442, 1444, 1466, 1483, 1489, 1494,
                  1557, 1564, 1585, 1592, 1601, 1610, 1622, 1626, 1638, 1652, 1656, 1660, 1665, 1673, 1681, 1682, 1691,
                  1698, 1699, 1730, 1732, 1738, 1760, 1816, 1818, 1823, 1836, 1862, 1865, 1883, 1884, 1893, 1899, 1902,
                  1962, 1984, 1989, 1995, 2001, 2044, 2055, 2072, 2088, 2096, 2110, 2168, 2175, 2183, 2203, 2210, 2234,
                  2249, 2260, 2309, 2330, 2352, 2357, 2381, 2402, 2406, 2421, 2424, 2445, 2485, 2553, 2564, 2582, 2589,
                  2608, 2610, 2630, 2632, 2660, 2709, 2717, 2731, 2737, 2787, 2814, 2819, 2836, 2875, 2877, 2878, 2881,
                  2882, 2883, 2886, 2891, 2893, 2895, 2896, 2899, 2901, 2905, 2906, 2907, 2909, 2912, 2913, 2917, 2923,
                  2926, 2928, 2930, 2933, 2936, 2939, 2940, 2942, 2945, 2946, 2965, 2981, 2983, 3007, 3011, 3047, 3055,
                  3062, 3092, 3111, 3184, 3201, 3248, 3310, 3324, 3337, 3366, 3390, 3404, 3409, 3415, 3429, 3459, 3511,
                  3575, 3602, 3628, 3691, 3706, 3710, 3839, 3960, 4089, 4104, 4138, 4169, 4201, 4204, 4317, 4330, 4446,
                  4470, 4483, 4629, 4649, 4692, 4726, 4877, 4937, 4951, 5001, 114, 118, 123, 203, 283, 286, 315, 325,
                  402, 417, 421, 424, 430, 462, 470, 481, 495, 508, 530, 556, 557, 568, 579, 589, 597, 612, 627, 656,
                  661, 669, 677, 688, 689, 715, 721, 733, 752, 761, 768, 780, 801, 820, 879, 887, 944, 952, 1004, 1013,
                  1021, 1025, 1041, 1066, 1075, 1096, 1154, 1172, 1210, 1232, 1240, 1251, 1255, 1267, 1299, 1305, 1376,
                  1383, 1397, 1422, 1427, 1453, 1464, 1471, 1479, 1480, 1500, 1509, 1513, 1569, 1590, 1615, 1642, 1644,
                  1689, 1712, 1713, 1740, 1742, 1757, 1811, 1845, 1855, 1863, 1895, 1903, 1919, 1988, 1997, 2006, 2064
                  ]
    '''

    '''
    for index in used_index:
        print(index, end=', ')
    
    import numpy as np
    record = np.zeros(num_classes)
    for index in range(len(dataset.ids)):
        img, target = dataset.__getitem__(index)
        for j in range(num_classes):
            if (target[:, -1] == j).sum() > 0:
                record[j] += 1#(target[:, -1] == j).sum()
    '''


    '''
    # 1-5 30 shots
    used_index = [3, 4, 12, 25, 31, 32, 37, 50, 53, 59, 64, 70, 74, 75, 76, 85, 87, 93, 99, 104, 109, 113, 114, 116,
                  121, 122, 129, 133, 142, 143, 149, 151, 152, 155, 156, 161, 171, 172, 173, 185, 188, 193, 194, 202,
                  207, 209, 212, 216, 217, 218, 219, 222, 229, 239, 247, 251, 269, 270, 283, 289, 292, 296, 301, 302,
                  312, 320, 331, 333, 335, 336, 337, 340, 341, 346, 348, 350, 351, 352, 359, 361, 362, 365, 369, 371,
                  373, 380, 381, 391, 392, 393, 396, 419, 433, 437, 446, 447, 473, 475, 478, 484, 492, 494, 499, 500,
                  501, 509, 515, 535, 538, 544, 545, 547, 556, 565, 567, 575, 582, 587, 591, 599, 600, 605, 608, 609,
                  610, 612, 627, 631, 632, 635, 649, 657, 663, 667, 671, 672, 682, 691, 693, 698, 699, 708, 710, 712,
                  713, 719, 721, 726, 727, 736]
    
    # 6-10 30 shots
    used_index = [23, 34, 37, 59, 102, 111, 138, 141, 154, 155, 158, 163, 166, 174, 179, 191, 205, 230, 235, 271, 281,
                  287, 301, 311, 331, 364, 367, 377, 391, 393, 395, 396, 405, 407, 409, 410, 414, 416, 417, 418, 419,
                  421, 425, 432, 434, 435, 436, 437, 441, 443, 446, 448, 451, 452, 453, 457, 458, 462, 465, 467, 470,
                  472, 474, 477, 478, 480, 481, 508, 517, 526, 533, 535, 542, 544, 561, 562, 567, 569, 576, 577, 587,
                  591, 601, 606, 609, 615, 617, 623, 636, 641, 645, 647, 652, 654, 659, 665, 666, 667, 670, 671, 672,
                  673, 676, 677, 678, 679, 688, 692, 698, 701, 704, 705, 707, 709, 714, 715, 717, 720, 722, 732, 738,
                  758, 766, 784, 794, 818, 830, 833, 839, 846, 853, 857, 873, 901, 942, 945, 959, 960, 985, 996, 997,
                  1018, 1025, 1048, 1062, 1085, 1086, 1113, 1137, 1143]

    # 11-15 30 shots
    used_index = [6, 46, 52, 58, 59, 61, 64, 65, 80, 92, 98, 100, 106, 112, 114, 123, 124, 143, 146, 152, 156, 186, 209,
                  213, 222, 236, 241, 247, 256, 263, 267, 269, 270, 282, 285, 290, 293, 294, 296, 306, 309, 310, 311,
                  312, 316, 317, 319, 321, 338, 345, 346, 361, 363, 367, 383, 390, 400, 403, 411, 422, 430, 432, 449,
                  451, 452, 457, 460, 462, 470, 486, 505, 513, 521, 525, 542, 543, 575, 578, 579, 584, 593, 596, 599,
                  603, 618, 639, 649, 656, 670, 680, 688, 692, 695, 706, 711, 729, 738, 744, 750, 759, 766, 774, 817,
                  826, 836, 846, 887, 890, 898, 907, 909, 914, 935, 951, 956, 958, 961, 987, 997, 1009, 1012, 1013,
                  1014, 1017, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034,
                  1035, 1036, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1046, 1047]

    # 16-20 30 shots
    used_index = [5, 6, 24, 30, 34, 40, 50, 54, 57, 63, 69, 70, 72, 77, 86, 89, 90, 91, 92, 96, 98, 102, 117, 118, 120,
                  121, 124, 125, 126, 128, 135, 144, 175, 176, 181, 183, 187, 188, 190, 202, 235, 239, 240, 248, 263,
                  284, 288, 289, 293, 297, 311, 313, 327, 330, 353, 356, 378, 406, 419, 424, 425, 429, 430, 436, 442,
                  446, 448, 449, 454, 457, 459, 461, 465, 467, 468, 471, 472, 475, 479, 488, 493, 499, 500, 513, 517,
                  518, 519, 521, 523, 525, 526, 527, 529, 531, 533, 534, 535, 541, 545, 550, 551, 557, 563, 570, 575,
                  579, 580, 585, 590, 600, 605, 607, 608, 616, 617, 619, 621, 625, 629, 633, 636, 638, 641, 644, 645,
                  646, 657, 667, 668, 670, 675, 677, 679, 686, 694, 698, 699, 701, 702, 704, 705, 713, 717, 721, 725,
                  726, 733, 735, 737, 739]
    '''

    # 1-5 1 shots
    # used_index = [3, 13, 16, 17, 19]
    #used_index = [50, 52, 54, 58, 62]
    #used_index = [104, 106, 111, 115, 119]

    # 6-10 1 shots
    # used_index = [23, 24, 26, 27, 41]
    # used_index = [59, 60, 75, 77, 87]
    # used_index = [102, 106, 110, 113, 134]

    # 11-15 1 shots
    # used_index = [6, 7, 31, 38, 39]
    #used_index = [52, 63, 87, 104, 106]
    #used_index = [100, 110, 117, 126, 129]

    # 16-20 1 shots
    used_index = [5, 11, 14, 19, 21]
    # used_index = [50, 61, 62, 64, 67]
    # used_index = [117, 135, 137, 140, 142]

    '''
    temp = []
    for index in used_index:
        temp.append(dataset.ids[index])
    dataset.ids = []
    dataset.ids = temp[:]

    
    trainval_temp = []
    for index in used_index:
        trainval_temp.append(dataset.ids[index][1])

    maindir = '/media/data2/xychen/LSCLIP-txt/'
    with open(maindir + "trainval.txt", "w") as f:
        for line in trainval_temp:
            f.write(line + "\n")
    '''

    temp = []
    for rep in range(10):
        for index in used_index:
            temp.append(dataset.ids[index])
    dataset.ids = []
    dataset.ids = temp[:]

    import numpy as np
    record_each = np.zeros(num_classes)
    for index in range(len(dataset.ids)):
        img, target = dataset.__getitem__(index)
        for j in range(num_classes):
            if (target[:, -1] == j).sum() > 0.01:
                record_each[j] += 1  # (target[:, -1] == j).sum()
    return dataset