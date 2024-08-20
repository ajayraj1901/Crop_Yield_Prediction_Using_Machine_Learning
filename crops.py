def crop(crop_name):
    crop_data = {
    "wheat":["/static/images/wheat.jpg", "BAGALKOT., BELGAUM, , DHARWAD, , GULBARGA", "rabi","Sri Lanka, United Arab Emirates, Taiwan"],
    "paddy":["/static/images/paddy.jpg", "BELGAUM, BELLARY, CHIKAMAGLUR, CHITHRADURGA, DAKSHIN KARNATAKA", "kharif","Bangladesh, Saudi Arabia, Iran"],
    "barley":["/static/images/barley.jpg", "BENGALURU URBAN, BELLARY, BELGAUM, CHARAMAJANAGAR, YADGIR", "rabi","Oman, UK, Qatar, USA"],
    "maize":["/static/images/maize.jpg", "CHARAMAJANAGAR, CHIKBALLAPUR, CHITHRADURGA, DAVANAGERE, HASSAN", "kharif", "Hong Kong, United Arab Emirates, France"],
    "bajra":["/static/images/bajra.jpg", "KOPPAL, KOLAR, GULBARGA, CHIKAMAGLUR", "kharif", "Oman, Saudi Arabia, Israel, Japan"],
    "copra":["/static/images/copra.jpg", "TUMKUR, DAVANAGERE, BELLARY, BELAGUAM, BENGALURU RURAL, UDUPI","rabi", "Veitnam, Bangladesh, Iran, Malaysia"],
    "cotton":["/static/images/cotton.jpg", "UDUPI, DAKSHIN KARNATAKA, YADGIR, KOLAR, CHITHRADURGA, HASSAN", " China, Bangladesh, Egypt"],
    "masoor":["/static/images/masoor.jpg", "SHIMOGA, KOPPAL, KOLAR, TUMKUR, BELLARY", "rabi", "Pakistan, Cyprus,United Arab Emirates"],
    "gram":["/static/images/gram.jpg", "CHARAMJANAGARA, CHIKBALLAPUR, HASSAN, UTTAR KARNATAKA, BELGAUM", "rabi", "Veitnam, Spain, Myanmar"],
    "groundnut":["/static/images/groundnut.jpg", "BANGALORE URBAN, KOPPAL, UDUPI, DAKSHIN KARNATAKA, SHIMOGA", "kharif", "Indonesia, Jordan, Iraq"],
    "arhar":["/static/images/arhar.jpg", "BAGALKOT, CHIKAMAGALUR, KOLLAR, DAVANAGERE", "kharif", "United Arab Emirates, USA, Chicago"],
    "sesamum":["/static/images/sesamum.jpg", "BAGALKOT, BELLARY, BELGAUM, KOLAR, YADGIR", "rabi", "Iraq, South Africa, USA, Netherlands"],
    "jowar":["/static/images/jowar.jpg", "CHIKAMAGALUR, CHARAMJANAGARA, GULBARGA, DHARWAD, GADAG", "kharif", "Torronto, Sydney, New York"],
    "moong":["/static/images/moong.jpg", "HAVERI, KODAGU, HASSAN", "rabi", "Qatar, United States, Canada"],
    "niger":["/static/images/niger.jpg", "BELGAUM, MANDYA, RAICHUR, RAMANAGARA, SHIMOGA", "kharif", "United States of American,Argenyina, Belgium"],
    "rape":["/static/images/rape.jpg", "BELLARY, HASSAN, KODAGU, HAVERI, KOPPAL", "rabi", "Veitnam, Malaysia, Taiwan"],
    "jute":["/static/images/jute.jpg", "YADGIR , RAICHUR , HASSAN , RAMANAGARA , YADGIR", "kharif", "JOrdan, United Arab Emirates, Taiwan"],
    "safflower":["/static/images/safflower.jpg",  "MANDYA, TUMKUR, BAGALKOT, CHIKAMAGALUR, BELLARY", "kharif", " Philippines, Taiwan, Portugal"],
    "soyabean":["/static/images/soyabean.jpg",  "BELGAUM, HASSAN, KOLAR, MANDYA", "kharif", "Spain, Thailand, Singapore"],
    "urad":["/static/images/urad.jpg",  "CHARAMAJANAGARA, GADAG, DAVANAGERE, TUMKUR", "rabi", "United States, Canada, United Arab Emirates"],
    "ragi":["/static/images/ragi.jpg",  "BELGAUM, SHIMOGA, HASSAN", "kharif", "United Arab Emirates, New Zealand, Bahrain"],
    "sunflower":["/static/images/sunflower.jpg",  "GADAG, RAICHUR, SHIMOGA, BELLARY, BANGALORE URBAN", "rabi", "Phillippines, United States, Bangladesh"],
    "sugarcane":["/static/images/sugarcane.jpg","UTTAR KARNATAKA, MANDYA, TUMKUR, BELLARY, BELGAUM" , "kharif", "Kenya, United Arab Emirates, United Kingdom"]
    }
    return crop_data[crop_name]