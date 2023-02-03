def day_to_date(d):
 	date = {
 	0: "01.01", 1: "02.01", 2: "03.01", 3: "04.01", 4: "05.01", 5: "06.01", 6: "07.01", 7: "08.01",\
 	8: "09.01", 9: "10.01", 10: "11.01", 11: "12.01", 12: "13.01", 13: "14.01", 14: "15.01", 15: "16.01",\
 	16: "17.01", 17: "18.01", 18: "19.01", 19: "20.01", 20: "21.01", 21: "22.01", 22: "23.01", 23: "24.01",\
 	24: "25.01", 25: "26.01", 26: "27.01", 27: "28.01", 28: "29.01", 29: "30.01", 30: "31.01", 31: "01.02",\
 	32: "02.02", 33: "03.02", 34: "04.02", 35: "05.02", 36: "06.02", 37: "07.02", 38: "08.02", 39: "09.02",\
 	40: "10.02", 41: "11.02", 42: "12.02", 43: "13.02", 44: "14.02", 45: "15.02", 46: "16.02", 47: "17.02",\
 	48: "18.02", 49: "19.02", 50: "20.02", 51: "21.02", 52: "22.02", 53: "23.02", 54: "24.02", 55: "25.02",\
 	56: "26.02", 57: "27.02", 58: "28.02", 59: "01.03", 60: "02.03", 61: "03.03", 62: "04.03", 63: "05.03",\
 	64: "06.03", 65: "07.03", 66: "08.03", 67: "09.03", 68: "10.03", 69: "11.03", 70: "12.03", 71: "13.03",\
 	72: "14.03", 73: "15.03", 74: "16.03", 75: "17.03", 76: "18.03", 77: "19.03", 78: "20.03", 79: "21.03",\
 	80: "22.03", 81: "23.03", 82: "24.03", 83: "25.03", 84: "26.03", 85: "27.03", 86: "28.03", 87: "29.03",\
 	88: "30.03", 89: "31.03", 90: "01.04", 91: "02.04", 92: "03.04", 93: "04.04", 94: "05.04", 95: "06.04",\
 	96: "07.04", 97: "08.04", 98: "09.04", 99: "10.04", 100: "11.04", 101: "12.04", 102: "13.04", 103: "14.04",\
 	104: "15.04", 105: "16.04", 106: "17.04", 107: "18.04", 108: "19.04", 109: "20.04", 110: "21.04", 111: "22.04",\
 	112: "23.04", 113: "24.04", 114: "25.04", 115: "26.04", 116: "27.04", 117: "28.04", 118: "29.04", 119: "30.04",\
 	120: "01.05", 121: "02.05", 122: "03.05", 123: "04.05", 124: "05.05", 125: "06.05", 126: "07.05", 127: "08.05",\
 	128: "09.05", 129: "10.05", 130: "11.05", 131: "12.05", 132: "13.05", 133: "14.05", 134: "15.05", 135: "16.05",\
 	136: "17.05", 137: "18.05", 138: "19.05", 139: "20.05", 140: "21.05", 141: "22.05", 142: "23.05", 143: "24.05",\
 	144: "25.05", 145: "26.05", 146: "27.05", 147: "28.05", 148: "29.05", 149: "30.05", 150: "31.05", 151: "01.06",\
 	152: "02.06", 153: "03.06", 154: "04.06", 155: "05.06", 156: "06.06", 157: "07.06", 158: "08.06", 159: "09.06",\
 	160: "10.06", 161: "11.06", 162: "12.06", 163: "13.06", 164: "14.06", 165: "15.06", 166: "16.06", 167: "17.06",\
 	168: "18.06", 169: "19.06", 170: "20.06", 171: "21.06", 172: "22.06", 172: "23.06", 174: "24.06", 175: "25.06",\
 	176: "26.06", 177: "27.06", 178: "28.06", 179: "29.06", 180: "30.06", 181: "01.07", 182: "02.07", 183: "03.07",\
 	184: "04.07", 185: "05.07", 186: "06.07", 187: "07.07", 188: "08.07", 189: "09.07", 190: "10.07", 191: "11.07",\
 	192: "12.07", 193: "13.07", 194: "14.07", 195: "15.07", 196: "16.07", 197: "17.07", 198: "18.07", 199: "19.07",\
 	200: "20.07", 201: "21.07", 202: "22.07", 203: "23.07", 204: "24.07", 205: "25.07", 206: "26.07", 207: "27.07",\
 	208: "28.07", 209: "29.07", 210: "30.07", 211: "31.07", 212: "01.08", 213: "02.08", 214: "03.08", 215: "04.08",\
 	216: "05.08", 217: "06.08", 218: "07.08", 219: "08.08", 220: "09.08", 221: "10.08", 222: "11.08", 223: "12.08",\
 	224: "13.08", 225: "14.08", 226: "15.08", 227: "16.08", 228: "17.08", 229: "18.08", 230: "19.08", 231: "20.08",\
 	232: "21.08", 233: "22.08", 234: "23.08", 235: "24.08", 236: "25.08", 237: "26.08", 238: "27.08", 239: "28.08",\
 	240: "29.08", 241: "30.08", 242: "31.08", 243: "01.09", 244: "02.09", 245: "03.09", 246: "04.09", 247: "05.09",\
 	248: "06.09", 249: "07.09", 250: "08.09", 251: "09.09", 252: "10.09", 253: "11.09", 254: "12.09", 255: "13.09",\
 	256: "14.09", 257: "15.09", 258: "16.09", 259: "17.09", 260: "18.09", 261: "19.09", 262: "20.09", 263: "21.09",\
 	264: "22.09", 265: "23.09", 266: "24.09", 267: "25.09", 268: "26.09", 269: "27.09", 270: "28.09", 271: "29.09",\
 	272: "30.09", 273: "01.10", 274: "02.10", 275: "03.10", 276: "04.10", 277: "05.10", 278: "06.10", 279: "07.10",\
 	280: "08.10", 281: "09.10", 282: "10.10", 283: "11.10", 284: "12.10", 285: "13.10", 286: "14.10", 287: "15.10",\
 	288: "16.10", 289: "17.10", 290: "18.10", 291: "19.10", 292: "20.10", 293: "21.10", 294: "22.10", 295: "23.10",\
 	296: "24.10", 297: "25.10", 298: "26.10", 299: "27.10", 300: "28.10", 301: "29.10", 302: "30.10", 303: "31.10",\
 	304: "01.11", 305: "02.11", 306: "03.11", 307: "04.11", 308: "05.11", 309: "06.11", 310: "07.11", 311: "08.11",\
 	312: "09.11", 313: "10.11", 314: "11.11", 315: "12.11", 316: "13.11", 317: "14.11", 318: "15.11", 319: "16.11",\
 	320: "17.11", 321: "18.11", 322: "19.11", 323: "20.11", 324: "21.11", 325: "22.11", 326: "23.11", 327: "24.11",\
 	328: "25.11", 329: "26.11", 330: "27.11", 331: "28.11", 332: "29.11", 333: "30.11", 334: "01.12", 335: "02.12",\
 	336: "03.12", 337: "04.12", 338: "05.12", 339: "06.12", 340: "07.12", 341: "08.12", 342: "09.12", 343: "10.12",\
 	344: "11.12", 345: "12.12", 346: "13.12", 347: "14.12", 348: "15.12", 349: "16.12", 350: "17.12", 351: "18.12",\
 	352: "19.12", 353: "20.12", 354: "21.12", 355: "22.12", 356: "23.12", 257: "24.12", 358: "25.12", 359: "26.12",\
 	360: "27.12", 361: "28.12", 362: "29.12", 363: "30.12", 364: "31.12"
 	}
 	return date[d]

def date_to_day(d):
	day = {
 	0: "01.01", 1: "02.01", 2: "03.01", 3: "04.01", 4: "05.01", 5: "06.01", 6: "07.01", 7: "08.01",\
 	8: "09.01", 9: "10.01", 10: "11.01", 11: "12.01", 12: "13.01", 13: "14.01", 14: "15.01", 15: "16.01",\
 	16: "17.01", 17: "18.01", 18: "19.01", 19: "20.01", 20: "21.01", 21: "22.01", 22: "23.01", 23: "24.01",\
 	24: "25.01", 25: "26.01", 26: "27.01", 27: "28.01", 28: "29.01", 29: "30.01", 30: "31.01", 31: "01.02",\
 	32: "02.02", 33: "03.02", 34: "04.02", 35: "05.02", 36: "06.02", 37: "07.02", 38: "08.02", 39: "09.02",\
 	40: "10.02", 41: "11.02", 42: "12.02", 43: "13.02", 44: "14.02", 45: "15.02", 46: "16.02", 47: "17.02",\
 	48: "18.02", 49: "19.02", 50: "20.02", 51: "21.02", 52: "22.02", 53: "23.02", 54: "24.02", 55: "25.02",\
 	56: "26.02", 57: "27.02", 58: "28.02", 59: "01.03", 60: "02.03", 61: "03.03", 62: "04.03", 63: "05.03",\
 	64: "06.03", 65: "07.03", 66: "08.03", 67: "09.03", 68: "10.03", 69: "11.03", 70: "12.03", 71: "13.03",\
 	72: "14.03", 73: "15.03", 74: "16.03", 75: "17.03", 76: "18.03", 77: "19.03", 78: "20.03", 79: "21.03",\
 	80: "22.03", 81: "23.03", 82: "24.03", 83: "25.03", 84: "26.03", 85: "27.03", 86: "28.03", 87: "29.03",\
 	88: "30.03", 89: "31.03", 90: "01.04", 91: "02.04", 92: "03.04", 93: "04.04", 94: "05.04", 95: "06.04",\
 	96: "07.04", 97: "08.04", 98: "09.04", 99: "10.04", 100: "11.04", 101: "12.04", 102: "13.04", 103: "14.04",\
 	104: "15.04", 105: "16.04", 106: "17.04", 107: "18.04", 108: "19.04", 109: "20.04", 110: "21.04", 111: "22.04",\
 	112: "23.04", 113: "24.04", 114: "25.04", 115: "26.04", 116: "27.04", 117: "28.04", 118: "29.04", 119: "30.04",\
 	120: "01.05", 121: "02.05", 122: "03.05", 123: "04.05", 124: "05.05", 125: "06.05", 126: "07.05", 127: "08.05",\
 	128: "09.05", 129: "10.05", 130: "11.05", 131: "12.05", 132: "13.05", 133: "14.05", 134: "15.05", 135: "16.05",\
 	136: "17.05", 137: "18.05", 138: "19.05", 139: "20.05", 140: "21.05", 141: "22.05", 142: "23.05", 143: "24.05",\
 	144: "25.05", 145: "26.05", 146: "27.05", 147: "28.05", 148: "29.05", 149: "30.05", 150: "31.05", 151: "01.06",\
 	152: "02.06", 153: "03.06", 154: "04.06", 155: "05.06", 156: "06.06", 157: "07.06", 158: "08.06", 159: "09.06",\
 	160: "10.06", 161: "11.06", 162: "12.06", 163: "13.06", 164: "14.06", 165: "15.06", 166: "16.06", 167: "17.06",\
 	168: "18.06", 169: "19.06", 170: "20.06", 171: "21.06", 172: "22.06", 172: "23.06", 174: "24.06", 175: "25.06",\
 	176: "26.06", 177: "27.06", 178: "28.06", 179: "29.06", 180: "30.06", 181: "01.07", 182: "02.07", 183: "03.07",\
 	184: "04.07", 185: "05.07", 186: "06.07", 187: "07.07", 188: "08.07", 189: "09.07", 190: "10.07", 191: "11.07",\
 	192: "12.07", 193: "13.07", 194: "14.07", 195: "15.07", 196: "16.07", 197: "17.07", 198: "18.07", 199: "19.07",\
 	200: "20.07", 201: "21.07", 202: "22.07", 203: "23.07", 204: "24.07", 205: "25.07", 206: "26.07", 207: "27.07",\
 	208: "28.07", 209: "29.07", 210: "30.07", 211: "31.07", 212: "01.08", 213: "02.08", 214: "03.08", 215: "04.08",\
 	216: "05.08", 217: "06.08", 218: "07.08", 219: "08.08", 220: "09.08", 221: "10.08", 222: "11.08", 223: "12.08",\
 	224: "13.08", 225: "14.08", 226: "15.08", 227: "16.08", 228: "17.08", 229: "18.08", 230: "19.08", 231: "20.08",\
 	232: "21.08", 233: "22.08", 234: "23.08", 235: "24.08", 236: "25.08", 237: "26.08", 238: "27.08", 239: "28.08",\
 	240: "29.08", 241: "30.08", 242: "31.08", 243: "01.09", 244: "02.09", 245: "03.09", 246: "04.09", 247: "05.09",\
 	248: "06.09", 249: "07.09", 250: "08.09", 251: "09.09", 252: "10.09", 253: "11.09", 254: "12.09", 255: "13.09",\
 	256: "14.09", 257: "15.09", 258: "16.09", 259: "17.09", 260: "18.09", 261: "19.09", 262: "20.09", 263: "21.09",\
 	264: "22.09", 265: "23.09", 266: "24.09", 267: "25.09", 268: "26.09", 269: "27.09", 270: "28.09", 271: "29.09",\
 	272: "30.09", 273: "01.10", 274: "02.10", 275: "03.10", 276: "04.10", 277: "05.10", 278: "06.10", 279: "07.10",\
 	280: "08.10", 281: "09.10", 282: "10.10", 283: "11.10", 284: "12.10", 285: "13.10", 286: "14.10", 287: "15.10",\
 	288: "16.10", 289: "17.10", 290: "18.10", 291: "19.10", 292: "20.10", 293: "21.10", 294: "22.10", 295: "23.10",\
 	296: "24.10", 297: "25.10", 298: "26.10", 299: "27.10", 300: "28.10", 301: "29.10", 302: "30.10", 303: "31.10",\
 	304: "01.11", 305: "02.11", 306: "03.11", 307: "04.11", 308: "05.11", 309: "06.11", 310: "07.11", 311: "08.11",\
 	312: "09.11", 313: "10.11", 314: "11.11", 315: "12.11", 316: "13.11", 317: "14.11", 318: "15.11", 319: "16.11",\
 	320: "17.11", 321: "18.11", 322: "19.11", 323: "20.11", 324: "21.11", 325: "22.11", 326: "23.11", 327: "24.11",\
 	328: "25.11", 329: "26.11", 330: "27.11", 331: "28.11", 332: "29.11", 333: "30.11", 334: "01.12", 335: "02.12",\
 	336: "03.12", 337: "04.12", 338: "05.12", 339: "06.12", 340: "07.12", 341: "08.12", 342: "09.12", 343: "10.12",\
 	344: "11.12", 345: "12.12", 346: "13.12", 347: "14.12", 348: "15.12", 349: "16.12", 350: "17.12", 351: "18.12",\
 	352: "19.12", 353: "20.12", 354: "21.12", 355: "22.12", 356: "23.12", 257: "24.12", 358: "25.12", 359: "26.12",\
 	360: "27.12", 361: "28.12", 362: "29.12", 363: "30.12", 364: "31.12"
 	}

	for key, value in day.items():
		if d == value:
			day = key
	return day


















